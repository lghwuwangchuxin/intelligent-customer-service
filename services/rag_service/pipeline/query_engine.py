"""
Enhanced RAG Query Engine / 增强型 RAG 查询引擎
===============================================

整合所有 RAG 组件的统一查询管道。

核心功能
--------
1. **查询转换**: HyDE + Query Expansion
2. **混合检索**: Vector + BM25 + RRF 融合
3. **重排序**: Jina Reranker
4. **后处理**: 语义去重 + MMR
5. **响应生成**: LLM 生成答案
6. **Langfuse 追踪**: 完整流程监控

Langfuse 追踪结构
-----------------
```
Trace: rag_query
├── Span: query_transform
│   ├── Generation: hyde
│   └── Generation: query_expansion
├── Span: hybrid_retrieval
│   ├── Span: vector_search
│   └── Span: bm25_search
├── Span: reranking
├── Span: post_processing
│   ├── deduplication
│   └── mmr_diversity
└── Generation: response_generation
```

Author: Intelligent Customer Service Team
Version: 2.0.0
"""
import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from llama_index.core import Settings as LlamaSettings
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode

from config.settings import settings
from app.rag.query_transform import QueryTransformer, get_query_transformer
from app.rag.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from app.rag.reranker import JinaReranker, get_reranker
from app.rag.postprocessor import RAGPostProcessor, get_post_processor
from app.rag.index_manager import IndexManager, get_index_manager
from app.rag.ragas_evaluator import get_rag_evaluator, RAGEvaluator

# Langfuse observability
from app.services.langfuse_service import get_langfuse_service

# Enhanced logging utilities
from app.utils.log_utils import (
    LogContext,
    log_phase_start,
    log_phase_end,
    log_step,
    log_substep,
    log_query_info,
    log_documents_retrieved,
    log_llm_call,
    log_scores_distribution,
    log_timing_summary,
    log_config,
    log_error_detail,
    log_warning_detail,
    log_pipeline_start,
    log_pipeline_stage,
    log_pipeline_end,
    RAGLogger,
    PerformanceLogger,
    SEPARATOR_LIGHT,
)

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG query engine."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "metadata": self.metadata,
        }


@dataclass
class RAGContext:
    """Context information for RAG generation."""
    documents: List[str]
    sources: List[Dict[str, Any]]
    scores: List[float]


# System prompt for RAG response generation
RAG_SYSTEM_PROMPT = """你是一个专业的智能客服助手。请根据以下检索到的相关文档内容来回答用户的问题。

重要规则:
1. 只使用提供的文档内容来回答问题
2. 如果文档中没有相关信息，请明确说明"根据现有知识库，暂无相关信息"
3. 回答要准确、简洁、专业
4. 不要编造或推测文档中没有的信息

检索到的相关文档:
{context}

请根据上述文档内容回答用户问题。"""


class EnhancedRAGQueryEngine:
    """
    Enhanced RAG query engine with full pipeline:
    1. Query transformation (HyDE, Expansion)
    2. Hybrid retrieval (Vector + BM25)
    3. Reranking (Jina)
    4. Post-processing (Dedup, MMR)
    5. Response generation
    """

    def __init__(
        self,
        index_manager: Optional[IndexManager] = None,
        query_transformer: Optional[QueryTransformer] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[JinaReranker] = None,
        post_processor: Optional[RAGPostProcessor] = None,
        evaluator: Optional[RAGEvaluator] = None,
    ):
        self.index_manager = index_manager or get_index_manager()
        self.query_transformer = query_transformer or get_query_transformer()
        self.hybrid_retriever = hybrid_retriever or get_hybrid_retriever()
        self.reranker = reranker or get_reranker()
        self.post_processor = post_processor or get_post_processor()
        self.evaluator = evaluator or get_rag_evaluator()

        self._llm = LlamaSettings.llm

        logger.info("[EnhancedRAG] Query engine initialized with all components (including evaluator)")

    async def aquery(
        self,
        question: str,
        user_id: str = None,
        session_id: str = None,
    ) -> RAGResponse:
        """
        Execute async RAG query with full pipeline and Langfuse tracing.

        Parameters
        ----------
        question : str
            User question

        user_id : str, optional
            User ID for Langfuse tracking

        session_id : str, optional
            Session ID for Langfuse tracking

        Returns
        -------
        RAGResponse
            Response with answer and sources
        """
        # 创建增强日志上下文
        import uuid
        request_id = str(uuid.uuid4())[:8]
        log_ctx = LogContext(module="RAG", request_id=request_id, user_id=user_id, session_id=session_id)
        rag_logger = RAGLogger(request_id=request_id)
        perf_logger = PerformanceLogger(module="RAG", request_id=request_id)

        # 定义流水线阶段
        pipeline_stages = ["QueryTransform", "HybridRetrieval", "Reranking", "PostProcessing", "ResponseGeneration"]

        # 记录查询开始和流水线信息
        rag_logger.start_query(question, user_id)
        log_pipeline_start(log_ctx, "RAG Query", pipeline_stages)

        # 记录配置信息
        log_config(log_ctx, "RAG Settings", {
            "HyDE": settings.RAG_ENABLE_HYDE,
            "Query Expansion": settings.RAG_ENABLE_QUERY_EXPANSION,
            "Expansion Num": settings.RAG_QUERY_EXPANSION_NUM if settings.RAG_ENABLE_QUERY_EXPANSION else "N/A",
            "Hybrid Search": settings.RAG_ENABLE_HYBRID,
            "Vector Weight": settings.RAG_VECTOR_WEIGHT if settings.RAG_ENABLE_HYBRID else "N/A",
            "BM25 Weight": settings.RAG_BM25_WEIGHT if settings.RAG_ENABLE_HYBRID else "N/A",
            "Rerank": settings.RAG_ENABLE_RERANK,
            "Rerank Model": settings.RAG_RERANK_MODEL if settings.RAG_ENABLE_RERANK else "N/A",
            "Rerank Top N": settings.RAG_RERANK_TOP_N if settings.RAG_ENABLE_RERANK else "N/A",
            "LLM Provider": settings.LLM_PROVIDER,
            "LLM Model": settings.LLM_MODEL,
        })

        perf_logger.start("RAG Query")

        # 创建 Langfuse 追踪
        langfuse = get_langfuse_service()
        trace = None
        if langfuse.enabled:
            trace = langfuse.create_trace(
                name="rag_query",
                user_id=user_id,
                session_id=session_id,
                input={"question": question},
                metadata={
                    "hyde_enabled": settings.RAG_ENABLE_HYDE,
                    "query_expansion_enabled": settings.RAG_ENABLE_QUERY_EXPANSION,
                    "hybrid_enabled": settings.RAG_ENABLE_HYBRID,
                    "rerank_enabled": settings.RAG_ENABLE_RERANK,
                },
                tags=["rag", "query"],
            )

        total_start_time = time.time()

        completed_stages = 0
        try:
            # Step 1: Query transformation
            log_pipeline_stage(log_ctx, "QueryTransform", 1, 5, "running")
            log_step(log_ctx, "Step 1", "查询转换 (HyDE + Query Expansion)")
            step1_start = time.time()
            perf_logger.checkpoint("query_transform_start")

            transform_span = None
            if trace:
                transform_span = langfuse.create_span(
                    trace,
                    name="query_transform",
                    input={"original_query": question},
                )

            queries = await self.query_transformer.transform(question)
            step1_elapsed = (time.time() - step1_start) * 1000

            # 记录转换结果详情
            rag_logger.log_transform(question, queries, step1_elapsed)
            log_step(log_ctx, "Step 1", f"生成 {len(queries)} 个查询变体", data={"queries": queries[:3]})
            log_pipeline_stage(log_ctx, "QueryTransform", 1, 5, "completed")
            perf_logger.checkpoint("query_transform_end")
            completed_stages += 1

            if transform_span:
                langfuse.end_span(
                    transform_span,
                    output={
                        "num_queries": len(queries),
                        "queries": queries,
                        "elapsed_seconds": round(step1_elapsed, 3),
                    },
                )

            # Step 2: Hybrid retrieval
            log_pipeline_stage(log_ctx, "HybridRetrieval", 2, 5, "running")
            log_step(log_ctx, "Step 2", f"混合检索 (Vector + BM25) - 处理 {len(queries)} 个查询")
            step2_start = time.time()
            perf_logger.checkpoint("retrieval_start")

            retrieval_span = None
            if trace:
                retrieval_span = langfuse.create_span(
                    trace,
                    name="hybrid_retrieval",
                    input={"num_queries": len(queries)},
                )

            nodes = await self._retrieve_multi_query(queries)
            step2_elapsed = (time.time() - step2_start) * 1000
            perf_logger.checkpoint("retrieval_end")

            # 记录检索结果详情
            if nodes:
                retrieval_scores = [n.score for n in nodes]
                rag_logger.log_retrieval("HybridRetrieval", len(nodes), retrieval_scores, step2_elapsed)
                log_documents_retrieved(log_ctx, len(nodes), "Hybrid", retrieval_scores[:5])
                log_scores_distribution(log_ctx, "Retrieval", retrieval_scores)
                log_pipeline_stage(log_ctx, "HybridRetrieval", 2, 5, "completed")
                completed_stages += 1
            else:
                log_step(log_ctx, "Step 2", "未检索到任何文档", level="warning")
                log_warning_detail(log_ctx, "检索结果为空", "EmptyRetrieval",
                                 "请检查知识库是否已正确索引，或尝试调整查询关键词")
                log_pipeline_stage(log_ctx, "HybridRetrieval", 2, 5, "failed")

            if retrieval_span:
                langfuse.end_span(
                    retrieval_span,
                    output={
                        "num_nodes": len(nodes),
                        "elapsed_seconds": round(step2_elapsed / 1000, 3),
                    },
                )

            if not nodes:
                logger.warning(f"[RAG] [{request_id}] 检索结果为空，无法继续处理")
                if trace:
                    langfuse.end_trace(
                        trace,
                        output={"status": "no_documents", "answer": ""},
                    )
                return RAGResponse(
                    answer="根据现有知识库，暂无与您问题相关的信息。请尝试用其他方式描述您的问题。",
                    sources=[],
                    metadata={"status": "no_documents"}
                )

            # Step 3: Reranking
            log_pipeline_stage(log_ctx, "Reranking", 3, 5, "running")
            nodes_before_rerank = len(nodes)
            log_step(log_ctx, "Step 3", f"神经重排序 (模型: {settings.RAG_RERANK_MODEL})")
            step3_start = time.time()
            perf_logger.checkpoint("rerank_start")

            rerank_span = None
            if trace:
                rerank_span = langfuse.create_span(
                    trace,
                    name="reranking",
                    input={"num_nodes_before": len(nodes)},
                    metadata={"model": settings.RAG_RERANK_MODEL},
                )

            query_bundle = QueryBundle(query_str=question)
            nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
            step3_elapsed = (time.time() - step3_start) * 1000
            perf_logger.checkpoint("rerank_end")

            # 记录重排序结果
            rerank_scores = [n.score for n in nodes]
            rag_logger.log_rerank(nodes_before_rerank, len(nodes), rerank_scores, step3_elapsed)
            log_step(log_ctx, "Step 3", f"重排序 {nodes_before_rerank} → {len(nodes)} 文档",
                     data={"top_scores": [round(s, 4) for s in rerank_scores[:3]]})
            log_scores_distribution(log_ctx, "Rerank", rerank_scores)
            log_pipeline_stage(log_ctx, "Reranking", 3, 5, "completed")
            completed_stages += 1

            if rerank_span:
                langfuse.end_span(
                    rerank_span,
                    output={
                        "num_nodes_after": len(nodes),
                        "top_scores": [round(n.score, 4) for n in nodes[:3]],
                        "elapsed_seconds": round(step3_elapsed / 1000, 3),
                    },
                )

            # Step 4: Post-processing
            log_pipeline_stage(log_ctx, "PostProcessing", 4, 5, "running")
            nodes_before_postproc = len(nodes)
            log_step(log_ctx, "Step 4", f"后处理 (语义去重 + MMR多样性过滤)")
            step4_start = time.time()
            perf_logger.checkpoint("postprocess_start")

            postproc_span = None
            if trace:
                postproc_span = langfuse.create_span(
                    trace,
                    name="post_processing",
                    input={"num_nodes_before": len(nodes)},
                )

            nodes = await self.post_processor.aprocess(nodes, question)
            step4_elapsed = (time.time() - step4_start) * 1000
            perf_logger.checkpoint("postprocess_end")

            # 记录后处理结果
            rag_logger.log_postprocess(nodes_before_postproc, len(nodes), step4_elapsed)
            log_step(log_ctx, "Step 4", f"后处理 {nodes_before_postproc} → {len(nodes)} 文档")
            log_pipeline_stage(log_ctx, "PostProcessing", 4, 5, "completed")
            completed_stages += 1

            if postproc_span:
                langfuse.end_span(
                    postproc_span,
                    output={
                        "num_nodes_after": len(nodes),
                        "elapsed_seconds": round(step4_elapsed / 1000, 3),
                    },
                )

            # Step 5: Build context and generate response
            log_pipeline_stage(log_ctx, "ResponseGeneration", 5, 5, "running")
            log_step(log_ctx, "Step 5", f"LLM 响应生成 (模型: {settings.LLM_PROVIDER}/{settings.LLM_MODEL})")
            step5_start = time.time()
            perf_logger.checkpoint("generation_start")

            context = self._build_context(nodes)

            # 记录上下文构建信息
            context_text = "\n".join(context.documents)
            log_substep(log_ctx, "Step 5", "Context",
                        f"构建上下文: {len(context.documents)} 文档, {len(context_text)} 字符")

            answer = await self._generate_response(question, context)
            step5_elapsed = (time.time() - step5_start) * 1000
            perf_logger.checkpoint("generation_end")

            # 记录 LLM 生成详情
            rag_logger.log_generation(len(context_text), len(answer), step5_elapsed)
            log_llm_call(log_ctx, "ResponseGen",
                        model=f"{settings.LLM_PROVIDER}/{settings.LLM_MODEL}",
                        input_tokens=len(context_text) // 4,
                        output_tokens=len(answer) // 4,
                        elapsed_ms=step5_elapsed)
            log_pipeline_stage(log_ctx, "ResponseGeneration", 5, 5, "completed")
            completed_stages += 1

            # 记录 Langfuse 生成
            if trace and langfuse.enabled:
                langfuse.log_generation(
                    trace=trace,
                    name="response_generation",
                    model=f"{settings.LLM_PROVIDER}/{settings.LLM_MODEL}",
                    input={"question": question, "context_length": len(context_text)},
                    output=answer,
                    usage={
                        "input_tokens": len(context_text) // 4,
                        "output_tokens": len(answer) // 4,
                    },
                    metadata={
                        "num_sources": len(context.sources),
                        "latency_seconds": round(step5_elapsed / 1000, 3),
                    },
                )

            total_elapsed = (time.time() - total_start_time) * 1000

            # 记录步骤耗时
            log_ctx.record_step_time("QueryTransform", step1_elapsed)
            log_ctx.record_step_time("HybridRetrieval", step2_elapsed)
            log_ctx.record_step_time("Reranking", step3_elapsed)
            log_ctx.record_step_time("PostProcessing", step4_elapsed)
            log_ctx.record_step_time("ResponseGeneration", step5_elapsed)

            # 输出耗时汇总
            log_timing_summary(log_ctx)

            # 记录查询完成和流水线结束
            rag_logger.end_query(True, len(answer), len(context.sources))
            log_pipeline_end(log_ctx, "RAG Query", success=True,
                           completed_stages=completed_stages, total_stages=5)
            perf_logger.end("RAG Query")
            logger.info(f"[RAG] [{request_id}] 查询完成 | 回答长度: {len(answer)} 字符 | 来源: {len(context.sources)} 个文档 | 总耗时: {total_elapsed:.0f}ms")

            # 结束追踪
            if trace:
                avg_score = sum(context.scores) / len(context.scores) if context.scores else 0
                langfuse.end_trace(
                    trace,
                    output={
                        "status": "success",
                        "answer": answer[:500],  # 截断避免太长
                        "num_sources": len(context.sources),
                    },
                    metadata={
                        "total_elapsed_seconds": round(total_elapsed, 3),
                        "step_times": {
                            "query_transform": round(step1_elapsed, 3),
                            "retrieval": round(step2_elapsed, 3),
                            "reranking": round(step3_elapsed, 3),
                            "post_processing": round(step4_elapsed, 3),
                            "generation": round(step5_elapsed, 3),
                        },
                        "avg_score": round(avg_score, 4),
                    },
                )

                # 记录检索质量评分
                langfuse.log_score(
                    trace=trace,
                    name="retrieval_relevance",
                    value=avg_score,
                    comment=f"Average retrieval score from {len(context.sources)} sources",
                )

            # Step 6: RAGAS Quality Evaluation (async, non-blocking)
            if settings.RAG_ENABLE_EVALUATION:
                import asyncio
                try:
                    # 异步执行 RAGAS 评估（不阻塞响应）
                    asyncio.create_task(self._async_ragas_evaluate(
                        question=question,
                        contexts=[s.get("content", "") for s in context.sources],
                        answer=answer,
                        trace=trace,
                        langfuse=langfuse,
                    ))
                except Exception as eval_error:
                    logger.warning(f"[RAGASEvaluator] 异步评估启动失败: {eval_error}")

            return RAGResponse(
                answer=answer,
                sources=context.sources,
                metadata={
                    "status": "success",
                    "num_queries": len(queries),
                    "num_sources": len(context.sources),
                    "avg_score": sum(context.scores) / len(context.scores) if context.scores else 0,
                    "total_time": round(total_elapsed, 3),
                }
            )

        except Exception as e:
            # 记录详细错误日志
            log_error_detail(log_ctx, e, "RAG Query",
                           context_data={
                               "question": question[:100],
                               "completed_stages": completed_stages,
                               "user_id": user_id,
                           })
            log_pipeline_end(log_ctx, "RAG Query", success=False,
                           completed_stages=completed_stages, total_stages=5)
            perf_logger.end("RAG Query")

            if trace:
                langfuse.end_trace(
                    trace,
                    output={"status": "error", "error": str(e)},
                    metadata={"error_type": type(e).__name__},
                )
            return RAGResponse(
                answer=f"抱歉，处理您的问题时出现了错误。请稍后重试。",
                sources=[],
                metadata={"status": "error", "error": str(e)}
            )

    def query(self, question: str) -> RAGResponse:
        """
        Execute sync RAG query with full pipeline.

        Args:
            question: User question

        Returns:
            RAGResponse with answer and sources
        """
        logger.info(f"[EnhancedRAG] Starting sync query: {question[:50]}...")

        try:
            # Step 1: Query transformation (sync)
            queries = self.query_transformer.transform_sync(question)

            # Step 2: Hybrid retrieval (sync)
            nodes = self._retrieve_multi_query_sync(queries)

            if not nodes:
                return RAGResponse(
                    answer="根据现有知识库，暂无与您问题相关的信息。",
                    sources=[],
                    metadata={"status": "no_documents"}
                )

            # Step 3: Reranking
            query_bundle = QueryBundle(query_str=question)
            nodes = self.reranker.postprocess_nodes(nodes, query_bundle)

            # Step 4: Post-processing
            nodes = self.post_processor.process(nodes, question)

            # Step 5: Generate response
            context = self._build_context(nodes)
            answer = self._generate_response_sync(question, context)

            return RAGResponse(
                answer=answer,
                sources=context.sources,
                metadata={
                    "status": "success",
                    "num_sources": len(context.sources),
                }
            )

        except Exception as e:
            logger.error(f"[EnhancedRAG] Query failed: {e}", exc_info=True)
            return RAGResponse(
                answer=f"抱歉，处理您的问题时出现了错误。",
                sources=[],
                metadata={"status": "error", "error": str(e)}
            )

    async def _retrieve_multi_query(self, queries: List[str]) -> List[NodeWithScore]:
        """
        Retrieve nodes for multiple queries using parallel async retrieval.

        优化: 使用 HybridRetriever 的 aretrieve_multi_query 方法实现并行检索
        """
        # 检索前先检查组件状态
        bm25_ready = self.hybrid_retriever.is_bm25_ready()
        bm25_stats = self.hybrid_retriever.get_bm25_stats()
        logger.info(
            f"[EnhancedRAG] 检索诊断 - BM25就绪: {bm25_ready}, "
            f"索引文档数: {bm25_stats.get('num_documents', 0)}, "
            f"混合检索: {settings.RAG_ENABLE_HYBRID}"
        )

        if not bm25_ready and settings.RAG_ENABLE_HYBRID:
            logger.warning(
                "[EnhancedRAG] ⚠️ BM25索引未就绪，混合检索将降级为纯向量检索。"
                "建议重新初始化知识库或重启服务。"
            )

        try:
            # 使用并行多查询检索 (内部已实现 RRF 融合和去重)
            nodes = await self.hybrid_retriever.aretrieve_multi_query(queries)

            # 检索结果诊断
            if not nodes:
                logger.warning(
                    f"[EnhancedRAG] ⚠️ 检索返回0个结果。诊断信息:\n"
                    f"  - 查询数量: {len(queries)}\n"
                    f"  - BM25状态: {bm25_stats}\n"
                    f"  - 建议: 检查知识库是否已索引，或尝试更通用的查询词"
                )
            else:
                logger.info(
                    f"[EnhancedRAG] 检索成功: {len(nodes)} 个结果, "
                    f"分数范围: [{min(n.score for n in nodes):.4f}, {max(n.score for n in nodes):.4f}]"
                )

            return nodes
        except Exception as e:
            logger.warning(f"[EnhancedRAG] Parallel retrieval failed, falling back: {e}")
            # 回退到串行检索
            all_nodes: List[NodeWithScore] = []
            for query in queries:
                bundle = QueryBundle(query_str=query)
                try:
                    nodes = await self.hybrid_retriever.aretrieve(bundle)
                    all_nodes.extend(nodes)
                except Exception as e2:
                    logger.warning(f"[EnhancedRAG] Retrieval failed for query: {e2}")

            # Deduplicate by node_id
            seen_ids = set()
            unique_nodes = []
            for node in all_nodes:
                if node.node.node_id not in seen_ids:
                    seen_ids.add(node.node.node_id)
                    unique_nodes.append(node)

            return unique_nodes

    def _retrieve_multi_query_sync(self, queries: List[str]) -> List[NodeWithScore]:
        """Sync version of multi-query retrieval."""
        return self.hybrid_retriever.retrieve_multi_query(queries)

    def _build_context(self, nodes: List[NodeWithScore]) -> RAGContext:
        """Build context from retrieved nodes."""
        documents = []
        sources = []
        scores = []

        for i, node in enumerate(nodes):
            content = node.node.get_content()
            documents.append(content)
            scores.append(node.score)

            # Extract source metadata
            metadata = node.node.metadata or {}
            sources.append({
                "index": i + 1,
                "content": content[:200] + "..." if len(content) > 200 else content,
                "score": round(node.score, 4),
                "source": metadata.get("source", "unknown"),
                "file_name": metadata.get("file_name", ""),
            })

        return RAGContext(
            documents=documents,
            sources=sources,
            scores=scores,
        )

    async def _generate_response(self, question: str, context: RAGContext) -> str:
        """Generate response using LLM."""
        # Format context
        context_str = "\n\n".join([
            f"[文档{i+1}]:\n{doc}"
            for i, doc in enumerate(context.documents)
        ])

        # Build prompt
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context_str)

        # Generate response
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        try:
            response = await self._llm.acomplete(
                f"{system_prompt}\n\n用户问题: {question}"
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"[EnhancedRAG] LLM generation failed: {e}")
            raise

    def _generate_response_sync(self, question: str, context: RAGContext) -> str:
        """Sync version of response generation."""
        context_str = "\n\n".join([
            f"[文档{i+1}]:\n{doc}"
            for i, doc in enumerate(context.documents)
        ])

        system_prompt = RAG_SYSTEM_PROMPT.format(context=context_str)

        response = self._llm.complete(
            f"{system_prompt}\n\n用户问题: {question}"
        )
        return response.text.strip()

    async def _async_ragas_evaluate(
        self,
        question: str,
        contexts: List[str],
        answer: str,
        trace=None,
        langfuse=None,
    ) -> None:
        """
        异步执行 LlamaIndex + RAGAS 质量评估（后台任务）。

        使用 LlamaIndex Evaluation 模块:
        - FaithfulnessEvaluator: 忠实度评估
        - RelevancyEvaluator: 相关性评估
        - CorrectnessEvaluator: 正确性评估

        可选 RAGAS 框架指标:
        - context_precision, context_recall
        - faithfulness, answer_relevancy
        """
        # 使用 ragas_evaluator 模块的 logger 以便日志正确路由到 ragas-eval-*.log
        ragas_logger = logging.getLogger('app.rag.ragas_evaluator')

        try:
            result = await self.evaluator.evaluate(
                query=question,
                response=answer,
                contexts=contexts,
                evaluate_retrieval=True,
                evaluate_generation=True,
            )

            # 记录评估结果（使用 ragas_logger 确保日志输出到正确的文件）
            relevancy = max(result.generation.relevancy, result.generation.answer_relevancy)
            ragas_logger.info(
                f"[RAGEvaluator] 异步评估完成 ({result.evaluator_type}) - "
                f"问题: {question[:50]}... | "
                f"综合分: {result.overall_score:.2f}, "
                f"忠实度: {result.generation.faithfulness:.2f}, "
                f"相关性: {relevancy:.2f}"
            )

            # 记录改进建议
            if result.suggestions:
                for suggestion in result.suggestions:
                    ragas_logger.info(f"[RAGEvaluator] 建议: {suggestion}")

            # 将评估分数记录到 Langfuse
            if trace and langfuse and langfuse.enabled:
                langfuse.log_score(
                    trace=trace,
                    name="rag_overall",
                    value=result.overall_score,
                    comment="RAG overall quality score (LlamaIndex + RAGAS)",
                )
                langfuse.log_score(
                    trace=trace,
                    name="rag_faithfulness",
                    value=result.generation.faithfulness,
                    comment="Answer faithfulness to context",
                )
                langfuse.log_score(
                    trace=trace,
                    name="rag_relevancy",
                    value=relevancy,
                    comment="Answer relevance to question",
                )

        except Exception as e:
            logger.warning(f"[RAGEvaluator] 后台评估失败: {e}")

    def retrieve_only(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents without generating a response.
        Useful for debugging and testing.

        Args:
            question: User question
            top_k: Number of results to return

        Returns:
            List of source documents with metadata
        """
        top_k = top_k or settings.RAG_FINAL_TOP_K
        logger.info(f"[EnhancedRAG] Retrieve-only query: {question[:50]}...")

        queries = self.query_transformer.transform_sync(question)
        nodes = self._retrieve_multi_query_sync(queries)

        query_bundle = QueryBundle(query_str=question)
        nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
        nodes = self.post_processor.process(nodes, question)

        context = self._build_context(nodes[:top_k])
        return context.sources


# Global singleton
_query_engine: Optional[EnhancedRAGQueryEngine] = None


def get_query_engine() -> EnhancedRAGQueryEngine:
    """Get the global query engine instance."""
    global _query_engine
    if _query_engine is None:
        _query_engine = EnhancedRAGQueryEngine()
    return _query_engine
