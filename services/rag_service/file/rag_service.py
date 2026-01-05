"""
RAG Service - Enhanced Retrieval Augmented Generation for customer service.
Refactored to use LlamaIndex-based EnhancedRAGQueryEngine with:
- Query transformation (HyDE, Expansion)
- Hybrid retrieval (Vector + BM25)
- Jina Reranker
- Post-processing (Dedup, MMR)
Integrated with Langfuse for observability.

Hybrid Storage Flow
-------------------
**Document Indexing:**
1. DocumentProcessor loads and chunks documents
2. HybridStoreManager generates embeddings
3. Stores text + metadata in Elasticsearch (for BM25)
4. Stores embedding + chunk_id in Milvus (for ANN)

**Retrieval:**
1. Query is vectorized
2. Milvus returns Top-K chunk_ids with semantic scores
3. ES BM25 returns chunk_ids with keyword scores
4. RRF fusion combines results
5. ES retrieves full text + metadata for final chunks
"""
import logging
import uuid
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from app.core.llm_manager import LLMManager
from app.core.vector_store import VectorStoreManager
from app.core.document_processor import DocumentProcessor
from app.core.langfuse_service import get_tracer
from app.core.hybrid_store_manager import get_hybrid_store_manager, HybridStoreManager
from app.domain.base.entities import DocumentChunk
from app.rag.query_engine import EnhancedRAGQueryEngine, get_query_engine, RAGResponse
from config.settings import settings

logger = logging.getLogger(__name__)


# System prompt for customer service (fallback for legacy mode)
CUSTOMER_SERVICE_PROMPT = """你是一个专业的智能客服助手。请根据以下知识库内容回答用户的问题。

知识库内容:
{context}

回答要求:
1. 基于知识库内容准确回答问题
2. 如果知识库中没有相关信息，诚实告知用户
3. 回答要简洁、专业、有礼貌
4. 如果需要，可以提供进一步的帮助建议

用户问题: {question}

请用中文回答:"""


class RAGService:
    """
    Enhanced RAG service for intelligent customer service.
    Uses LlamaIndex-based EnhancedRAGQueryEngine for improved retrieval.
    Falls back to LangChain LCEL chain when enhanced mode is disabled.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        vector_store_manager: VectorStoreManager,
        document_processor: DocumentProcessor,
        top_k: int = 5,
        use_enhanced_rag: bool = True,
    ):
        self.llm_manager = llm_manager
        self.vector_store = vector_store_manager
        self.document_processor = document_processor
        self.top_k = top_k
        self.use_enhanced_rag = use_enhanced_rag

        # Enhanced RAG query engine (lazy initialization)
        self._enhanced_engine: Optional[EnhancedRAGQueryEngine] = None

        # Legacy LCEL chain (for fallback)
        self._chain = self._build_chain()

        logger.info(
            f"[RAG] Service initialized - enhanced_rag: {use_enhanced_rag}, "
            f"top_k: {top_k}"
        )

    @property
    def enhanced_engine(self) -> EnhancedRAGQueryEngine:
        """Get or create the enhanced RAG query engine."""
        if self._enhanced_engine is None:
            logger.info("[RAG] Initializing EnhancedRAGQueryEngine")
            self._enhanced_engine = get_query_engine()
        return self._enhanced_engine

    def _build_chain(self):
        """Build the legacy RAG chain using LCEL."""
        prompt = ChatPromptTemplate.from_template(CUSTOMER_SERVICE_PROMPT)

        retriever = self.vector_store.get_retriever(
            search_kwargs={"k": self.top_k}
        )

        def format_docs(docs: List[Document]) -> str:
            """Format retrieved documents as context."""
            if not docs:
                return "暂无相关知识库内容。"
            return "\n\n".join(
                f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
                for doc in docs
            )

        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm_manager.llm
            | StrOutputParser()
        )

        return chain

    def query(self, question: str, session_id: Optional[str] = None) -> str:
        """
        Query the RAG system with Langfuse tracing.

        Args:
            question: User's question.
            session_id: Optional session ID for tracing.

        Returns:
            Generated answer.
        """
        logger.info(
            f"[RAG] 开始同步查询 - 问题: {question[:50]}..., "
            f"enhanced: {self.use_enhanced_rag}"
        )
        tracer = get_tracer()
        trace_id = session_id or str(uuid.uuid4())

        try:
            with tracer.trace(
                name="rag_query",
                session_id=trace_id,
                metadata={
                    "question": question,
                    "top_k": self.top_k,
                    "enhanced_rag": self.use_enhanced_rag,
                },
                tags=["rag", "sync", "enhanced" if self.use_enhanced_rag else "legacy"],
            ) as trace:
                if self.use_enhanced_rag:
                    response = self._enhanced_query(question, trace)
                else:
                    response = self._legacy_query(question, trace)

                logger.info(f"[RAG] 查询完成 - 响应长度: {len(response)}")
                trace.update(output=response)
                return response

        except Exception as e:
            logger.error(f"[RAG] 查询失败: {e}", exc_info=True)
            return "抱歉，处理您的问题时出现了错误。请稍后再试。"

    def _enhanced_query(self, question: str, trace) -> str:
        """Execute query using enhanced RAG pipeline."""
        logger.info("[RAG] 使用增强 RAG 管道")

        with trace.span(
            name="enhanced_rag_pipeline",
            input={"question": question},
        ) as span:
            # Query transformation
            with trace.span(name="query_transform") as qt_span:
                queries = self.enhanced_engine.query_transformer.transform_sync(question)
                qt_span.end(output={"num_queries": len(queries), "queries": queries})
                logger.info(f"[RAG] 查询转换完成 - 生成 {len(queries)} 个查询")

            # Retrieval
            with trace.span(name="hybrid_retrieval") as ret_span:
                result = self.enhanced_engine.query(question)
                ret_span.end(output={
                    "num_sources": len(result.sources),
                    "sources": [s.get("source", "") for s in result.sources],
                })
                logger.info(f"[RAG] 检索完成 - 获取 {len(result.sources)} 个来源")

            span.end(output={"answer_length": len(result.answer)})
            return result.answer

    def _legacy_query(self, question: str, trace) -> str:
        """Execute query using legacy LCEL chain."""
        logger.info("[RAG] 使用传统 LCEL 管道")

        with trace.span(
            name="retrieval",
            input={"query": question, "top_k": self.top_k},
        ) as retrieval_span:
            docs = self.vector_store.similarity_search(question, k=self.top_k)
            sources = [d.metadata.get("source", "unknown") for d in docs]
            retrieval_span.end(output={"doc_count": len(docs), "sources": sources})

        with trace.generation(
            name="llm_generation",
            model=f"{self.llm_manager.provider}/{self.llm_manager.model}",
            input=question,
        ) as gen:
            response = self._chain.invoke(question)
            gen.end(output=response)

        return response

    async def aquery(self, question: str, session_id: Optional[str] = None) -> str:
        """
        Async query the RAG system with Langfuse tracing.

        Args:
            question: User's question.
            session_id: Optional session ID for tracing.

        Returns:
            Generated answer.
        """
        logger.info(
            f"[RAG] 开始异步查询 - 问题: {question[:50]}..., "
            f"enhanced: {self.use_enhanced_rag}"
        )
        tracer = get_tracer()
        trace_id = session_id or str(uuid.uuid4())

        try:
            with tracer.trace(
                name="rag_query",
                session_id=trace_id,
                metadata={
                    "question": question,
                    "top_k": self.top_k,
                    "enhanced_rag": self.use_enhanced_rag,
                },
                tags=["rag", "async", "enhanced" if self.use_enhanced_rag else "legacy"],
            ) as trace:
                if self.use_enhanced_rag:
                    response = await self._enhanced_aquery(question, trace)
                else:
                    response = await self._legacy_aquery(question, trace)

                logger.info(f"[RAG] 异步查询完成 - 响应长度: {len(response)}")
                trace.update(output=response)
                return response

        except Exception as e:
            logger.error(f"[RAG] 异步查询失败: {e}", exc_info=True)
            return "抱歉，处理您的问题时出现了错误。请稍后再试。"

    async def _enhanced_aquery(self, question: str, trace) -> str:
        """Execute async query using enhanced RAG pipeline."""
        logger.info("[RAG] 使用增强 RAG 管道 (异步)")

        with trace.span(
            name="enhanced_rag_pipeline",
            input={"question": question},
        ) as span:
            # Full async pipeline
            result = await self.enhanced_engine.aquery(question)

            span.end(output={
                "answer_length": len(result.answer),
                "num_sources": len(result.sources),
                "status": result.metadata.get("status", "unknown"),
            })

            logger.info(
                f"[RAG] 增强管道完成 - 状态: {result.metadata.get('status')}, "
                f"来源数: {len(result.sources)}"
            )

            return result.answer

    async def _legacy_aquery(self, question: str, trace) -> str:
        """Execute async query using legacy LCEL chain."""
        logger.info("[RAG] 使用传统 LCEL 管道 (异步)")

        with trace.span(
            name="retrieval",
            input={"query": question, "top_k": self.top_k},
        ) as retrieval_span:
            docs = self.vector_store.similarity_search(question, k=self.top_k)
            sources = [d.metadata.get("source", "unknown") for d in docs]
            retrieval_span.end(output={"doc_count": len(docs), "sources": sources})

        with trace.generation(
            name="llm_generation",
            model=f"{self.llm_manager.provider}/{self.llm_manager.model}",
            input=question,
        ) as gen:
            response = await self._chain.ainvoke(question)
            gen.end(output=response)

        return response

    def stream_query(
        self,
        question: str,
        session_id: Optional[str] = None
    ) -> Iterator[str]:
        """
        Stream query response with Langfuse tracing.
        Note: Enhanced RAG does not support streaming, falls back to legacy.

        Args:
            question: User's question.
            session_id: Optional session ID for tracing.

        Yields:
            Response chunks.
        """
        logger.info(f"[RAG] 开始流式查询 - 问题: {question[:50]}...")
        tracer = get_tracer()
        trace_id = session_id or str(uuid.uuid4())
        full_response = []

        try:
            with tracer.trace(
                name="rag_stream_query",
                session_id=trace_id,
                metadata={"question": question, "top_k": self.top_k},
                tags=["rag", "stream"],
            ) as trace:
                with trace.span(
                    name="retrieval",
                    input={"query": question, "top_k": self.top_k},
                ) as retrieval_span:
                    docs = self.vector_store.similarity_search(question, k=self.top_k)
                    retrieval_span.end(
                        output={
                            "doc_count": len(docs),
                            "sources": [d.metadata.get("source", "unknown") for d in docs],
                        }
                    )

                with trace.generation(
                    name="llm_stream_generation",
                    model=f"{self.llm_manager.provider}/{self.llm_manager.model}",
                    input=question,
                ) as gen:
                    for chunk in self._chain.stream(question):
                        full_response.append(chunk)
                        yield chunk
                    gen.end(output="".join(full_response))

                trace.update(output="".join(full_response))

        except Exception as e:
            logger.error(f"[RAG] 流式查询失败: {e}")
            yield "抱歉，处理您的问题时出现了错误。"

    async def astream_query(
        self,
        question: str,
        session_id: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Async stream query response with Langfuse tracing.

        Args:
            question: User's question.
            session_id: Optional session ID for tracing.

        Yields:
            Response chunks.
        """
        logger.info(f"[RAG] 开始异步流式查询 - 问题: {question[:50]}...")
        tracer = get_tracer()
        trace_id = session_id or str(uuid.uuid4())
        full_response = []

        try:
            with tracer.trace(
                name="rag_stream_query",
                session_id=trace_id,
                metadata={"question": question, "top_k": self.top_k},
                tags=["rag", "async", "stream"],
            ) as trace:
                with trace.span(
                    name="retrieval",
                    input={"query": question, "top_k": self.top_k},
                ) as retrieval_span:
                    docs = self.vector_store.similarity_search(question, k=self.top_k)
                    retrieval_span.end(
                        output={
                            "doc_count": len(docs),
                            "sources": [d.metadata.get("source", "unknown") for d in docs],
                        }
                    )

                with trace.generation(
                    name="llm_stream_generation",
                    model=f"{self.llm_manager.provider}/{self.llm_manager.model}",
                    input=question,
                ) as gen:
                    async for chunk in self._chain.astream(question):
                        full_response.append(chunk)
                        yield chunk
                    gen.end(output="".join(full_response))

                trace.update(output="".join(full_response))

        except Exception as e:
            logger.error(f"[RAG] 异步流式查询失败: {e}")
            yield "抱歉，处理您的问题时出现了错误。"

    def get_relevant_documents(self, question: str) -> List[Dict[str, Any]]:
        """
        Get relevant documents without generating response.
        Uses enhanced RAG if enabled.

        Args:
            question: User's question.

        Returns:
            List of relevant document info.
        """
        logger.info(f"[RAG] 获取相关文档 - 问题: {question[:50]}...")

        try:
            if self.use_enhanced_rag:
                sources = self.enhanced_engine.retrieve_only(question, self.top_k)
                return [
                    {
                        "content": s.get("content", ""),
                        "source": s.get("source", "unknown"),
                        "score": s.get("score", 0),
                        "metadata": s,
                    }
                    for s in sources
                ]
            else:
                docs = self.vector_store.similarity_search(question, k=self.top_k)
                return [
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "metadata": doc.metadata,
                    }
                    for doc in docs
                ]
        except Exception as e:
            logger.error(f"[RAG] 获取相关文档失败: {e}")
            return []

    def add_knowledge(
        self,
        file_path: Optional[str] = None,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add knowledge to the knowledge base using hybrid storage.

        Hybrid Storage Flow:
        1. DocumentProcessor loads and chunks documents into TextNodes
        2. Convert TextNodes to DocumentChunks with unique chunk_ids
        3. HybridStoreManager indexes to both ES and Milvus:
           - ES: text + metadata (for BM25 search)
           - Milvus: embedding + chunk_id (for vector search)
        4. Update in-memory BM25 index for HybridRetriever

        Args:
            file_path: Path to file to add.
            text: Raw text to add.
            metadata: Optional metadata.

        Returns:
            Result dict with status.
        """
        import asyncio
        from app.rag.index_manager import get_index_manager
        from app.rag.hybrid_retriever import get_hybrid_retriever

        logger.info(f"[RAG] 添加知识 - file: {file_path}, text_len: {len(text) if text else 0}")

        try:
            # Step 1: Process document using DocumentProcessor
            if file_path:
                nodes = self.document_processor.process_file(file_path)
            elif text:
                nodes = self.document_processor.process_text(text, metadata)
            else:
                return {"success": False, "error": "No content provided"}

            if not nodes:
                return {"success": False, "error": "No content extracted from document"}

            # Step 2: Try hybrid storage first (ES + Milvus)
            hybrid_store = get_hybrid_store_manager()
            es_indexed = 0
            milvus_indexed = 0

            if hybrid_store and settings.HYBRID_STORAGE_ENABLED:
                # Convert TextNodes to DocumentChunks for HybridStoreManager
                doc_id = str(uuid.uuid4())
                chunks = []
                for i, node in enumerate(nodes):
                    chunk = DocumentChunk(
                        chunk_id=node.node_id or str(uuid.uuid4()),
                        doc_id=doc_id,
                        content=node.get_content(),
                        metadata={
                            **node.metadata,
                            "source": file_path or "text_input",
                        },
                        chunk_index=i,
                        chunk_total=len(nodes),
                    )
                    chunks.append(chunk)

                # Index to both ES and Milvus via HybridStoreManager
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(
                    hybrid_store.index_chunks(chunks, generate_embeddings=True)
                )
                es_indexed = result.get("es_indexed", 0)
                milvus_indexed = result.get("milvus_indexed", 0)

                logger.info(f"[RAG] Hybrid storage indexed: ES={es_indexed}, Milvus={milvus_indexed}")
            else:
                # Fallback: Use IndexManager for Milvus only
                logger.info("[RAG] Hybrid storage not available, using IndexManager only")
                index_manager = get_index_manager()
                index_manager.add_nodes(nodes, show_progress=True)
                milvus_indexed = len(nodes)

            # Step 3: Update in-memory BM25 index for HybridRetriever
            hybrid_retriever = get_hybrid_retriever()
            existing_nodes = hybrid_retriever._corpus_nodes.copy() if hybrid_retriever._corpus_nodes else []
            all_nodes = existing_nodes + nodes
            hybrid_retriever.build_bm25_index(all_nodes)

            # Step 4: Also sync to LangChain vector store for legacy compatibility
            documents = self.document_processor.to_langchain_documents(nodes)
            ids = self.vector_store.add_documents(documents)

            logger.info(
                f"[RAG] 知识添加成功 - 节点数: {len(nodes)}, "
                f"ES索引: {es_indexed}, Milvus索引: {milvus_indexed}"
            )
            return {
                "success": True,
                "num_documents": len(documents),
                "num_nodes": len(nodes),
                "ids": ids,
                "es_indexed": es_indexed,
                "milvus_indexed": milvus_indexed,
                "hybrid_storage": hybrid_store is not None,
            }
        except Exception as e:
            logger.error(f"[RAG] 添加知识失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def async_add_knowledge(
        self,
        file_path: Optional[str] = None,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Async add knowledge with hybrid storage (ES + Milvus).

        Hybrid Storage Flow:
        1. DocumentProcessor loads and chunks documents
        2. HybridStoreManager generates embeddings asynchronously
        3. Stores to ES (text + metadata) and Milvus (embedding + chunk_id) in parallel
        4. Updates in-memory BM25 index

        Features:
        - Async batch embedding for better performance
        - Text preprocessing for better LLM understanding
        - Dual storage: ES for BM25, Milvus for vector search

        Args:
            file_path: Path to file to add.
            text: Raw text to add.
            metadata: Optional metadata.

        Returns:
            Result dict with status.
        """
        import time
        from app.rag.index_manager import get_index_manager
        from app.rag.hybrid_retriever import get_hybrid_retriever

        start_time = time.time()
        logger.info(f"[RAG] 异步添加知识 - file: {file_path}, text_len: {len(text) if text else 0}")

        try:
            # Step 1: Process document using DocumentProcessor
            if file_path:
                nodes = self.document_processor.process_file(file_path)
            elif text:
                nodes = self.document_processor.process_text(text, metadata)
            else:
                return {"success": False, "error": "No content provided"}

            if not nodes:
                return {"success": False, "error": "No content extracted"}

            # Step 2: Try hybrid storage first (ES + Milvus)
            hybrid_store = get_hybrid_store_manager()
            es_indexed = 0
            milvus_indexed = 0

            if hybrid_store and settings.HYBRID_STORAGE_ENABLED:
                # Convert TextNodes to DocumentChunks for HybridStoreManager
                doc_id = str(uuid.uuid4())
                chunks = []
                for i, node in enumerate(nodes):
                    chunk = DocumentChunk(
                        chunk_id=node.node_id or str(uuid.uuid4()),
                        doc_id=doc_id,
                        content=node.get_content(),
                        metadata={
                            **node.metadata,
                            "source": file_path or "text_input",
                        },
                        chunk_index=i,
                        chunk_total=len(nodes),
                    )
                    chunks.append(chunk)

                # Async index to both ES and Milvus via HybridStoreManager
                result = await hybrid_store.index_chunks(
                    chunks,
                    batch_size=settings.EMBEDDING_BATCH_SIZE,
                    generate_embeddings=True,
                )
                es_indexed = result.get("es_indexed", 0)
                milvus_indexed = result.get("milvus_indexed", 0)

                logger.info(f"[RAG] Hybrid storage indexed: ES={es_indexed}, Milvus={milvus_indexed}")
            else:
                # Fallback: Use IndexManager for Milvus only (with async optimization)
                logger.info("[RAG] Hybrid storage not available, using IndexManager only")
                index_manager = get_index_manager()
                await index_manager.add_nodes_async(
                    nodes,
                    batch_size=settings.EMBEDDING_BATCH_SIZE,
                    max_concurrent=settings.EMBEDDING_MAX_CONCURRENT,
                    show_progress=True,
                    skip_duplicates=True,
                    delete_existing_source=True,
                    preprocess_text=True,
                )
                milvus_indexed = len(nodes)

            # Step 3: Update in-memory BM25 index for HybridRetriever
            hybrid_retriever = get_hybrid_retriever()
            existing_nodes = hybrid_retriever._corpus_nodes.copy() if hybrid_retriever._corpus_nodes else []
            all_nodes = existing_nodes + nodes
            hybrid_retriever.build_bm25_index(all_nodes)

            elapsed = time.time() - start_time
            speed = len(nodes) / elapsed if elapsed > 0 else 0

            logger.info(
                f"[RAG] 异步知识添加成功 - 节点数: {len(nodes)}, "
                f"ES索引: {es_indexed}, Milvus索引: {milvus_indexed}, "
                f"耗时: {elapsed:.2f}s, 速度: {speed:.1f} nodes/sec"
            )

            return {
                "success": True,
                "num_nodes": len(nodes),
                "es_indexed": es_indexed,
                "milvus_indexed": milvus_indexed,
                "hybrid_storage": hybrid_store is not None,
                "elapsed_seconds": round(elapsed, 2),
                "speed": round(speed, 1),
            }
        except Exception as e:
            logger.error(f"[RAG] 异步添加知识失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def index_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Index all documents in a directory (sync version).

        使用 LlamaIndex DocumentProcessor 批量处理目录下所有文档，
        同时更新向量索引和 BM25 索引。

        Note: For better performance, use async_index_directory() instead.

        Args:
            directory_path: Path to directory.

        Returns:
            Result dict with status.
        """
        import time
        from app.rag.index_manager import get_index_manager
        from app.rag.hybrid_retriever import get_hybrid_retriever

        start_time = time.time()

        logger.info(f"[RAG] ========================================")
        logger.info(f"[RAG] 开始索引目录: {directory_path}")
        logger.info(f"[RAG] ========================================")

        try:
            # 阶段0: 清除旧索引（确保使用新的 schema）
            logger.info(f"[RAG] 🗑️ 阶段0: 清除旧索引...")
            index_manager = get_index_manager()
            index_manager.clear_index()
            logger.info(f"[RAG] ✅ 旧索引已清除")

            # 阶段1: 文档加载和分块
            logger.info(f"[RAG] 📂 阶段1: 加载和分块文档...")
            nodes = self.document_processor.process_directory(directory_path)
            if not nodes:
                logger.warning(f"[RAG] ⚠️ 未找到任何文档")
                return {"success": False, "error": "No documents found"}

            logger.info(f"[RAG] ✅ 加载完成: {len(nodes)} 个节点")

            # 阶段2: 添加到 LlamaIndex 向量索引（供 EnhancedRAGQueryEngine 使用）
            logger.info(f"[RAG] 💾 阶段2: 添加到 LlamaIndex 向量索引...")
            node_ids = index_manager.add_nodes(nodes, show_progress=True)
            logger.info(f"[RAG] ✅ 向量索引完成: {len(node_ids)} 个节点已索引")

            # 阶段3: 构建 BM25 索引（供混合检索使用）
            logger.info(f"[RAG] 🔍 阶段3: 构建 BM25 索引...")
            hybrid_retriever = get_hybrid_retriever()
            hybrid_retriever.build_bm25_index(nodes)
            logger.info(f"[RAG] ✅ BM25 索引完成")

            elapsed = time.time() - start_time
            logger.info(f"[RAG] ========================================")
            logger.info(f"[RAG] ✅ 目录索引完成!")
            logger.info(f"[RAG] 📊 统计: {len(nodes)} 个节点")
            logger.info(f"[RAG] 📊 向量索引: ✓, BM25索引: ✓")
            logger.info(f"[RAG] ⏱️ 总耗时: {elapsed:.2f} 秒")
            logger.info(f"[RAG] ========================================")

            return {
                "success": True,
                "num_nodes": len(nodes),
                "ids": node_ids,
                "vector_index": True,
                "bm25_index": True,
                "elapsed_seconds": round(elapsed, 2),
            }
        except Exception as e:
            logger.error(f"[RAG] ❌ 索引目录失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def async_index_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Async index all documents in a directory with batch optimization.

        使用批量异步嵌入优化，显著提高索引速度。

        Args:
            directory_path: Path to directory.

        Returns:
            Result dict with status.
        """
        import time
        from app.rag.index_manager import get_index_manager
        from app.rag.hybrid_retriever import get_hybrid_retriever

        start_time = time.time()

        logger.info(f"[RAG] ========================================")
        logger.info(f"[RAG] 开始异步索引目录: {directory_path}")
        logger.info(f"[RAG] 批量大小: {settings.EMBEDDING_BATCH_SIZE}")
        logger.info(f"[RAG] 最大并发: {settings.EMBEDDING_MAX_CONCURRENT}")
        logger.info(f"[RAG] ========================================")

        try:
            # 阶段0: 清除旧索引
            logger.info(f"[RAG] 🗑️ 阶段0: 清除旧索引...")
            index_manager = get_index_manager()
            index_manager.clear_index()
            logger.info(f"[RAG] ✅ 旧索引已清除")

            # 阶段1: 文档加载和分块
            logger.info(f"[RAG] 📂 阶段1: 加载和分块文档...")
            nodes = self.document_processor.process_directory(directory_path)
            if not nodes:
                logger.warning(f"[RAG] ⚠️ 未找到任何文档")
                return {"success": False, "error": "No documents found"}

            logger.info(f"[RAG] ✅ 加载完成: {len(nodes)} 个节点")

            # 阶段2: 异步批量添加到向量索引
            logger.info(f"[RAG] 💾 阶段2: 异步批量添加到向量索引...")
            node_ids = await index_manager.add_nodes_async(
                nodes,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                max_concurrent=settings.EMBEDDING_MAX_CONCURRENT,
                show_progress=True,
            )
            logger.info(f"[RAG] ✅ 向量索引完成: {len(node_ids)} 个节点已索引")

            # 阶段2.5: 同步到 Elasticsearch
            logger.info(f"[RAG] 📝 阶段2.5: 同步到 Elasticsearch...")
            es_indexed = 0
            try:
                from app.core.hybrid_store_manager import get_hybrid_store_manager
                from app.core.elasticsearch_manager import DocumentChunk

                hybrid_store = get_hybrid_store_manager()
                if hybrid_store and hybrid_store.es_manager:
                    # 转换 TextNode 为 DocumentChunk
                    chunks = []
                    for i, node in enumerate(nodes):
                        chunk = DocumentChunk(
                            chunk_id=node.node_id,
                            doc_id=node.metadata.get("doc_id", node.metadata.get("file_name", "")),
                            content=node.text,
                            metadata={
                                "file_path": node.metadata.get("file_path", ""),
                                "file_name": node.metadata.get("file_name", ""),
                                "file_type": node.metadata.get("file_type", ""),
                                "file_size": node.metadata.get("file_size", 0),
                                "creation_date": node.metadata.get("creation_date", ""),
                                "last_modified_date": node.metadata.get("last_modified_date", ""),
                                "source": node.metadata.get("source", ""),
                            },
                            chunk_index=i,
                            chunk_total=len(nodes),
                            embedding=node.embedding if hasattr(node, 'embedding') else None,
                        )
                        chunks.append(chunk)

                    # 清除旧的 ES 索引数据
                    try:
                        await hybrid_store.es_manager.delete_all()
                        logger.info(f"[RAG] 🗑️ ES 旧数据已清除")
                    except Exception as e:
                        logger.warning(f"[RAG] ⚠️ 清除 ES 旧数据失败: {e}")

                    # 批量索引到 ES
                    result = await hybrid_store.es_manager.bulk_index_chunks(chunks, batch_size=500)
                    es_indexed = result.get("success", 0)
                    logger.info(f"[RAG] ✅ ES 索引完成: {es_indexed} 个文档")
                else:
                    logger.warning(f"[RAG] ⚠️ ES manager 不可用，跳过 ES 同步")
            except Exception as e:
                logger.error(f"[RAG] ❌ ES 同步失败: {e}", exc_info=True)

            # 阶段3: 构建 BM25 索引
            logger.info(f"[RAG] 🔍 阶段3: 构建 BM25 索引...")
            hybrid_retriever = get_hybrid_retriever()
            hybrid_retriever.build_bm25_index(nodes)
            logger.info(f"[RAG] ✅ BM25 索引完成")

            elapsed = time.time() - start_time
            speed = len(nodes) / elapsed if elapsed > 0 else 0

            logger.info(f"[RAG] ========================================")
            logger.info(f"[RAG] ✅ 异步索引完成!")
            logger.info(f"[RAG] 📊 统计: {len(nodes)} 个节点")
            logger.info(f"[RAG] 📊 向量索引(Milvus): ✓, ES索引: {es_indexed}, BM25索引: ✓")
            logger.info(f"[RAG] ⏱️ 总耗时: {elapsed:.2f} 秒 ({speed:.1f} 节点/秒)")
            logger.info(f"[RAG] ========================================")

            return {
                "success": True,
                "num_nodes": len(nodes),
                "ids": node_ids,
                "vector_index": True,
                "es_index": True,
                "es_indexed_count": es_indexed,
                "bm25_index": True,
                "elapsed_seconds": round(elapsed, 2),
                "nodes_per_second": round(speed, 1),
            }
        except Exception as e:
            logger.error(f"[RAG] ❌ 异步索引目录失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG pipeline."""
        stats = {
            "mode": "enhanced" if self.use_enhanced_rag else "legacy",
            "top_k": self.top_k,
            "llm_provider": self.llm_manager.provider,
            "llm_model": self.llm_manager.model,
        }

        if self.use_enhanced_rag:
            stats.update({
                "hyde_enabled": settings.RAG_ENABLE_HYDE,
                "query_expansion_enabled": settings.RAG_ENABLE_QUERY_EXPANSION,
                "hybrid_retrieval_enabled": settings.RAG_ENABLE_HYBRID,
                "rerank_enabled": settings.RAG_ENABLE_RERANK,
                "rerank_model": settings.RAG_RERANK_MODEL,
                "dedup_enabled": settings.RAG_ENABLE_DEDUP,
            })

        return stats


class ChatService:
    """
    Chat service for direct LLM conversations without RAG.
    Integrated with Langfuse for observability.
    """

    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.system_prompt = """你是一个友好、专业的智能客服助手。
请用简洁、有礼貌的方式回答用户的问题。如果不确定答案，请诚实告知。"""

    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Chat with the LLM with Langfuse tracing.

        Args:
            message: User message.
            history: Optional conversation history.
            session_id: Optional session ID for tracing.

        Returns:
            Assistant response.
        """
        logger.info(
            f"[Chat] 开始对话 - 消息: {message[:50]}..., "
            f"历史消息数: {len(history) if history else 0}"
        )
        tracer = get_tracer()
        trace_id = session_id or str(uuid.uuid4())

        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        with tracer.trace(
            name="chat",
            session_id=trace_id,
            metadata={"message": message, "history_length": len(history) if history else 0},
            tags=["chat", "sync"],
        ) as trace:
            with trace.generation(
                name="llm_chat",
                model=f"{self.llm_manager.provider}/{self.llm_manager.model}",
                input=messages,
            ) as gen:
                response = self.llm_manager.invoke(messages)
                gen.end(output=response)

            logger.info(f"[Chat] LLM 响应完成 - 响应长度: {len(response)}")
            trace.update(output=response)
            return response

    async def achat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Async chat with the LLM."""
        logger.info(
            f"[Chat] 开始异步对话 - 消息: {message[:50]}..., "
            f"历史消息数: {len(history) if history else 0}"
        )
        tracer = get_tracer()
        trace_id = session_id or str(uuid.uuid4())

        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        with tracer.trace(
            name="chat",
            session_id=trace_id,
            metadata={"message": message, "history_length": len(history) if history else 0},
            tags=["chat", "async"],
        ) as trace:
            with trace.generation(
                name="llm_chat",
                model=f"{self.llm_manager.provider}/{self.llm_manager.model}",
                input=messages,
            ) as gen:
                response = await self.llm_manager.ainvoke(messages)
                gen.end(output=response)

            logger.info(f"[Chat] LLM 异步响应完成 - 响应长度: {len(response)}")
            trace.update(output=response)
            return response

    def stream_chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream chat response with Langfuse tracing."""
        tracer = get_tracer()
        trace_id = session_id or str(uuid.uuid4())
        full_response = []

        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        with tracer.trace(
            name="chat_stream",
            session_id=trace_id,
            metadata={"message": message, "history_length": len(history) if history else 0},
            tags=["chat", "stream"],
        ) as trace:
            with trace.generation(
                name="llm_chat_stream",
                model=f"{self.llm_manager.provider}/{self.llm_manager.model}",
                input=messages,
            ) as gen:
                for chunk in self.llm_manager.stream(messages):
                    full_response.append(chunk)
                    yield chunk
                gen.end(output="".join(full_response))

            trace.update(output="".join(full_response))

    async def astream_chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Async stream chat response."""
        tracer = get_tracer()
        trace_id = session_id or str(uuid.uuid4())
        full_response = []

        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        with tracer.trace(
            name="chat_stream",
            session_id=trace_id,
            metadata={"message": message, "history_length": len(history) if history else 0},
            tags=["chat", "async", "stream"],
        ) as trace:
            with trace.generation(
                name="llm_chat_stream",
                model=f"{self.llm_manager.provider}/{self.llm_manager.model}",
                input=messages,
            ) as gen:
                async for chunk in self.llm_manager.astream(messages):
                    full_response.append(chunk)
                    yield chunk
                gen.end(output="".join(full_response))

            trace.update(output="".join(full_response))
