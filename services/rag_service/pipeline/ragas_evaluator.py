"""
RAG Quality Evaluator - LlamaIndex + RAGAS Integration
=======================================================

基于 LlamaIndex Evaluation 模块和 RAGAS 框架的 RAG 质量评估系统。

核心组件
--------
1. **LlamaIndex Evaluators**:
   - RetrieverEvaluator: 检索质量评估 (MRR, Hit Rate)
   - FaithfulnessEvaluator: 答案忠实度评估
   - RelevancyEvaluator: 答案相关性评估
   - CorrectnessEvaluator: 答案正确性评估
   - DatasetGenerator: 测试数据集生成

2. **RAGAS Metrics** (可选，需安装 ragas):
   - context_precision: 上下文精确度
   - context_recall: 上下文召回率
   - faithfulness: 忠实度
   - answer_relevancy: 答案相关性

评估维度
--------
| 维度 | LlamaIndex | RAGAS |
|------|------------|-------|
| 检索质量 | RetrieverEvaluator | context_precision, context_recall |
| 忠实度 | FaithfulnessEvaluator | faithfulness |
| 相关性 | RelevancyEvaluator | answer_relevancy |
| 正确性 | CorrectnessEvaluator | answer_correctness |

使用方法
--------
```python
from app.rag.ragas_evaluator import get_rag_evaluator

evaluator = get_rag_evaluator()

# 单次评估
result = await evaluator.evaluate(
    query="支付方式有哪些？",
    response="我们支持微信支付、支付宝和银行卡。",
    contexts=["我们支持微信支付、支付宝、银行卡..."],
    reference="支持微信支付、支付宝、银行卡支付"  # 可选
)

print(f"Faithfulness: {result.faithfulness}")
print(f"Relevancy: {result.relevancy}")
```

Version: 2.0.0
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional, Sequence
from dataclasses import dataclass, field, asdict
from datetime import datetime

from llama_index.core import Settings as LlamaSettings
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    BatchEvalRunner,
    EvaluationResult,
)
from llama_index.core.schema import TextNode
from llama_index.core.llms import LLM

from config.settings import settings

logger = logging.getLogger(__name__)

# Check RAGAS availability
RAGAS_AVAILABLE = False
try:
    from ragas import evaluate as ragas_evaluate
    # RAGAS 0.4.x uses ragas.metrics._faithfulness etc.
    try:
        # Try new import path first (RAGAS 0.4+)
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._context_recall import ContextRecall
        from ragas.metrics._answer_correctness import AnswerCorrectness
        faithfulness = Faithfulness()
        answer_relevancy = AnswerRelevancy()
        context_precision = ContextPrecision()
        context_recall = ContextRecall()
        answer_correctness = AnswerCorrectness()
    except ImportError:
        # Fallback to old import path (RAGAS < 0.4)
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
    logger.info("[RAGEvaluator] RAGAS framework detected and enabled (hybrid mode)")
except ImportError as e:
    logger.info(f"[RAGEvaluator] RAGAS not available ({e}), using LlamaIndex evaluators only")


# ============== Data Classes ==============

@dataclass
class RetrievalMetrics:
    """检索质量指标"""
    hit_rate: float = 0.0           # 命中率
    mrr: float = 0.0                # Mean Reciprocal Rank
    context_precision: float = 0.0  # 上下文精确度 (RAGAS)
    context_recall: float = 0.0     # 上下文召回率 (RAGAS)
    num_contexts: int = 0
    avg_context_score: float = 0.0


@dataclass
class GenerationMetrics:
    """生成质量指标"""
    faithfulness: float = 0.0       # 忠实度
    relevancy: float = 0.0          # 相关性
    correctness: float = 0.0        # 正确性 (需要 reference)
    answer_relevancy: float = 0.0   # 答案相关性 (RAGAS)

    # 详细信息
    faithfulness_feedback: str = ""
    relevancy_feedback: str = ""
    correctness_feedback: str = ""


@dataclass
class RAGEvaluationResult:
    """RAG 评估结果"""
    query: str
    response: str
    contexts: List[str]
    reference: Optional[str] = None

    # 检索指标
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)

    # 生成指标
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)

    # 综合评分
    overall_score: float = 0.0

    # 元数据
    evaluation_time_ms: int = 0
    evaluator_type: str = "llamaindex"  # "llamaindex" or "ragas" or "hybrid"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response[:200] + "..." if len(self.response) > 200 else self.response,
            "reference": self.reference,
            "retrieval": asdict(self.retrieval),
            "generation": asdict(self.generation),
            "overall_score": round(self.overall_score, 4),
            "evaluation_time_ms": self.evaluation_time_ms,
            "evaluator_type": self.evaluator_type,
            "timestamp": self.timestamp,
            "suggestions": self.suggestions,
        }


# ============== LlamaIndex Evaluator ==============

class LlamaIndexEvaluator:
    """
    基于 LlamaIndex 的 RAG 评估器。

    使用 LlamaIndex 官方评估模块:
    - FaithfulnessEvaluator: 评估答案是否忠实于上下文
    - RelevancyEvaluator: 评估答案与问题的相关性
    - CorrectnessEvaluator: 评估答案的正确性
    """

    def __init__(self, llm: Optional[LLM] = None):
        self._llm = llm or LlamaSettings.llm

        # Initialize evaluators
        self.faithfulness_evaluator = FaithfulnessEvaluator(llm=self._llm)
        self.relevancy_evaluator = RelevancyEvaluator(llm=self._llm)
        self.correctness_evaluator = CorrectnessEvaluator(llm=self._llm)

        logger.info("[LlamaIndexEvaluator] Initialized with FaithfulnessEvaluator, RelevancyEvaluator, CorrectnessEvaluator")

    async def evaluate_faithfulness(
        self,
        query: str,
        response: str,
        contexts: List[str],
    ) -> EvaluationResult:
        """评估答案忠实度"""
        try:
            result = await self.faithfulness_evaluator.aevaluate(
                query=query,
                response=response,
                contexts=contexts,
            )
            return result
        except Exception as e:
            logger.warning(f"[LlamaIndexEvaluator] Faithfulness evaluation failed: {e}")
            return EvaluationResult(query=query, response=response, passing=False, score=0.0, feedback=str(e))

    async def evaluate_relevancy(
        self,
        query: str,
        response: str,
        contexts: List[str],
    ) -> EvaluationResult:
        """评估答案相关性"""
        try:
            result = await self.relevancy_evaluator.aevaluate(
                query=query,
                response=response,
                contexts=contexts,
            )
            return result
        except Exception as e:
            logger.warning(f"[LlamaIndexEvaluator] Relevancy evaluation failed: {e}")
            return EvaluationResult(query=query, response=response, passing=False, score=0.0, feedback=str(e))

    async def evaluate_correctness(
        self,
        query: str,
        response: str,
        reference: str,
    ) -> EvaluationResult:
        """评估答案正确性（需要参考答案）"""
        try:
            result = await self.correctness_evaluator.aevaluate(
                query=query,
                response=response,
                reference=reference,
            )
            return result
        except Exception as e:
            logger.warning(f"[LlamaIndexEvaluator] Correctness evaluation failed: {e}")
            return EvaluationResult(query=query, response=response, passing=False, score=0.0, feedback=str(e))

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str],
        reference: Optional[str] = None,
    ) -> GenerationMetrics:
        """执行完整评估"""
        metrics = GenerationMetrics()

        # 并行执行评估
        tasks = [
            self.evaluate_faithfulness(query, response, contexts),
            self.evaluate_relevancy(query, response, contexts),
        ]

        if reference:
            tasks.append(self.evaluate_correctness(query, response, reference))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理忠实度结果
        if len(results) > 0 and isinstance(results[0], EvaluationResult):
            # 规范化分数到 [0, 1] 范围
            raw_score = results[0].score or 0.0
            metrics.faithfulness = min(1.0, max(0.0, raw_score))
            metrics.faithfulness_feedback = results[0].feedback or ""

        # 处理相关性结果
        if len(results) > 1 and isinstance(results[1], EvaluationResult):
            raw_score = results[1].score or 0.0
            metrics.relevancy = min(1.0, max(0.0, raw_score))
            metrics.relevancy_feedback = results[1].feedback or ""

        # 处理正确性结果
        if len(results) > 2 and isinstance(results[2], EvaluationResult):
            raw_score = results[2].score or 0.0
            # LlamaIndex CorrectnessEvaluator 可能返回大于1的分数，需要规范化
            # 如果分数在 0-5 范围，则归一化到 0-1
            if raw_score > 1.0:
                metrics.correctness = min(1.0, raw_score / 5.0)
            else:
                metrics.correctness = max(0.0, raw_score)
            metrics.correctness_feedback = results[2].feedback or ""

        return metrics


# ============== RAGAS Evaluator (Optional) ==============

class RAGASMetricsEvaluator:
    """
    基于 RAGAS 框架的评估器。

    RAGAS 指标:
    - faithfulness: 答案是否忠实于上下文
    - answer_relevancy: 答案与问题的相关性
    - context_precision: 上下文的精确度
    - context_recall: 上下文的召回率
    - answer_correctness: 答案的正确性

    注意: RAGAS 需要配置 LLM 后端。默认尝试使用 Ollama，如果失败则禁用 RAGAS 评估。
    """

    def __init__(self):
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS is not installed. Run: pip install ragas")

        self._llm = None
        self._embeddings = None
        self._initialized = False

        # 尝试配置 RAGAS 使用 Ollama
        try:
            self._configure_ollama_backend()
        except Exception as e:
            logger.warning(f"[RAGASEvaluator] Failed to configure Ollama backend: {e}")
            logger.info("[RAGASEvaluator] RAGAS will use default LLM (requires OPENAI_API_KEY)")

        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        logger.info("[RAGASEvaluator] Initialized with RAGAS metrics")

    def _configure_ollama_backend(self):
        """配置 RAGAS 使用 Ollama 作为 LLM 后端"""
        try:
            from langchain_ollama import ChatOllama, OllamaEmbeddings

            # 使用配置的 LLM 设置（支持 Ollama）
            # LLM_BASE_URL 格式: http://localhost:11434
            ollama_base_url = settings.LLM_BASE_URL
            model_name = settings.LLM_MODEL

            self._llm = ChatOllama(
                model=model_name,
                base_url=ollama_base_url,
            )

            self._embeddings = OllamaEmbeddings(
                model=settings.EMBEDDING_MODEL,
                base_url=settings.EMBEDDING_BASE_URL,
            )

            self._initialized = True
            logger.info(f"[RAGASEvaluator] Configured with Ollama backend: {model_name} @ {ollama_base_url}")

        except ImportError:
            logger.warning("[RAGASEvaluator] langchain_ollama not installed. Run: pip install langchain-ollama")
            raise
        except Exception as e:
            logger.warning(f"[RAGASEvaluator] Ollama configuration failed: {e}")
            raise

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str],
        reference: Optional[str] = None,
    ) -> Dict[str, float]:
        """执行 RAGAS 评估"""
        try:
            # 准备数据集
            data = {
                "question": [query],
                "answer": [response],
                "contexts": [contexts],
            }

            if reference:
                data["ground_truth"] = [reference]
                # 添加 answer_correctness 指标
                metrics = self.metrics + [answer_correctness]
            else:
                metrics = self.metrics

            dataset = Dataset.from_dict(data)

            # 构建评估参数
            eval_kwargs = {
                "dataset": dataset,
                "metrics": metrics,
            }

            # 如果配置了 Ollama，传入 LLM 和 Embeddings
            if self._initialized and self._llm and self._embeddings:
                eval_kwargs["llm"] = self._llm
                eval_kwargs["embeddings"] = self._embeddings

            # 执行评估（在线程中运行以避免阻塞）
            result = await asyncio.to_thread(
                ragas_evaluate,
                **eval_kwargs,
            )

            return dict(result)

        except Exception as e:
            logger.warning(f"[RAGASEvaluator] Evaluation failed: {e}")
            return {}


# ============== Unified RAG Evaluator ==============

class RAGEvaluator:
    """
    统一的 RAG 质量评估器。

    整合 LlamaIndex Evaluation 和 RAGAS 框架，提供:
    - 检索质量评估
    - 生成质量评估
    - 批量评估
    - 评估历史和统计

    Example:
        evaluator = RAGEvaluator()
        result = await evaluator.evaluate(
            query="如何退款？",
            response="您可以通过...",
            contexts=["退款流程..."],
        )
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        use_ragas: bool = True,
    ):
        """
        初始化评估器。

        Args:
            llm: LLM 实例，用于评估
            use_ragas: 是否使用 RAGAS（如果可用）
        """
        self.llm = llm
        self.use_ragas = use_ragas and RAGAS_AVAILABLE

        # 初始化 LlamaIndex 评估器
        self._llamaindex_evaluator = None

        # 初始化 RAGAS 评估器
        self._ragas_evaluator = None
        if self.use_ragas:
            try:
                self._ragas_evaluator = RAGASMetricsEvaluator()
            except Exception as e:
                logger.warning(f"[RAGEvaluator] Failed to initialize RAGAS: {e}")
                self.use_ragas = False

        # 评估历史
        self._evaluation_history: List[RAGEvaluationResult] = []

        evaluator_type = "hybrid (LlamaIndex + RAGAS)" if self.use_ragas else "LlamaIndex only"
        logger.info(f"[RAGEvaluator] Initialized with {evaluator_type}")

    @property
    def llamaindex_evaluator(self) -> LlamaIndexEvaluator:
        """延迟初始化 LlamaIndex 评估器"""
        if self._llamaindex_evaluator is None:
            self._llamaindex_evaluator = LlamaIndexEvaluator(llm=self.llm)
        return self._llamaindex_evaluator

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str],
        reference: Optional[str] = None,
        evaluate_retrieval: bool = True,
        evaluate_generation: bool = True,
    ) -> RAGEvaluationResult:
        """
        执行完整的 RAG 质量评估。

        Args:
            query: 用户问题
            response: 生成的答案
            contexts: 检索到的上下文列表
            reference: 参考答案（可选，用于正确性评估）
            evaluate_retrieval: 是否评估检索质量
            evaluate_generation: 是否评估生成质量

        Returns:
            RAGEvaluationResult: 评估结果
        """
        import time
        start_time = time.time()

        result = RAGEvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            reference=reference,
            retrieval=RetrievalMetrics(num_contexts=len(contexts)),
        )

        try:
            tasks = []

            # LlamaIndex 生成质量评估
            if evaluate_generation:
                tasks.append(self._evaluate_with_llamaindex(query, response, contexts, reference))

            # RAGAS 评估（如果可用）
            if self.use_ragas and self._ragas_evaluator:
                tasks.append(self._evaluate_with_ragas(query, response, contexts, reference))

            # 并行执行评估
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, res in enumerate(results):
                    if isinstance(res, Exception):
                        logger.warning(f"[RAGEvaluator] Evaluation task {i} failed: {res}")
                        continue

                    if isinstance(res, GenerationMetrics):
                        result.generation = res
                        result.evaluator_type = "llamaindex"
                    elif isinstance(res, dict):
                        # RAGAS 结果
                        self._merge_ragas_results(result, res)
                        result.evaluator_type = "hybrid" if result.generation.faithfulness > 0 else "ragas"

            # 计算综合评分
            result.overall_score = self._calculate_overall_score(result)

            # 生成改进建议
            result.suggestions = self._generate_suggestions(result)

        except Exception as e:
            logger.error(f"[RAGEvaluator] Evaluation failed: {e}")
            result.suggestions.append(f"评估过程出错: {str(e)}")

        result.evaluation_time_ms = int((time.time() - start_time) * 1000)

        # 记录历史
        self._evaluation_history.append(result)

        # 日志
        self._log_result(result)

        return result

    async def _evaluate_with_llamaindex(
        self,
        query: str,
        response: str,
        contexts: List[str],
        reference: Optional[str],
    ) -> GenerationMetrics:
        """使用 LlamaIndex 评估"""
        return await self.llamaindex_evaluator.evaluate(
            query=query,
            response=response,
            contexts=contexts,
            reference=reference,
        )

    async def _evaluate_with_ragas(
        self,
        query: str,
        response: str,
        contexts: List[str],
        reference: Optional[str],
    ) -> Dict[str, float]:
        """使用 RAGAS 评估"""
        if self._ragas_evaluator:
            return await self._ragas_evaluator.evaluate(
                query=query,
                response=response,
                contexts=contexts,
                reference=reference,
            )
        return {}

    def _merge_ragas_results(self, result: RAGEvaluationResult, ragas_scores: Dict[str, float]) -> None:
        """合并 RAGAS 评估结果"""
        def normalize_score(score: float) -> float:
            """规范化分数到 [0, 1] 范围"""
            if score is None or not isinstance(score, (int, float)):
                return 0.0
            # 如果分数大于1，可能是 0-5 或 0-100 的尺度
            if score > 1.0:
                if score <= 5.0:
                    return score / 5.0
                elif score <= 100.0:
                    return score / 100.0
                else:
                    return 1.0
            return max(0.0, min(1.0, score))

        # 检索指标
        if "context_precision" in ragas_scores:
            result.retrieval.context_precision = normalize_score(ragas_scores["context_precision"])
        if "context_recall" in ragas_scores:
            result.retrieval.context_recall = normalize_score(ragas_scores["context_recall"])

        # 生成指标
        if "faithfulness" in ragas_scores:
            # 如果 LlamaIndex 没有评估或分数为0，使用 RAGAS 的值
            if result.generation.faithfulness == 0:
                result.generation.faithfulness = normalize_score(ragas_scores["faithfulness"])
        if "answer_relevancy" in ragas_scores:
            result.generation.answer_relevancy = normalize_score(ragas_scores["answer_relevancy"])
        if "answer_correctness" in ragas_scores:
            if result.generation.correctness == 0:
                result.generation.correctness = normalize_score(ragas_scores["answer_correctness"])

    def _calculate_overall_score(self, result: RAGEvaluationResult) -> float:
        """计算综合评分，确保结果在 [0, 1] 范围内"""
        scores = []
        weights = []

        def clamp(v: float) -> float:
            """确保分数在 [0, 1] 范围内"""
            return max(0.0, min(1.0, v))

        # 检索质量 (20%)
        ctx_precision = clamp(result.retrieval.context_precision)
        ctx_recall = clamp(result.retrieval.context_recall)
        if ctx_precision > 0:
            scores.append(ctx_precision)
            weights.append(0.1)
        if ctx_recall > 0:
            scores.append(ctx_recall)
            weights.append(0.1)

        # 生成质量 (80%)
        faithfulness = clamp(result.generation.faithfulness)
        if faithfulness > 0:
            scores.append(faithfulness)
            weights.append(0.3)

        relevancy = clamp(max(result.generation.relevancy, result.generation.answer_relevancy))
        if relevancy > 0:
            scores.append(relevancy)
            weights.append(0.25)

        correctness = clamp(result.generation.correctness)
        if correctness > 0:
            scores.append(correctness)
            weights.append(0.15)

        if not scores:
            return 0.0

        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        overall = weighted_sum / total_weight if total_weight > 0 else 0.0

        # 最终确保综合分数在 [0, 1] 范围内
        return round(max(0.0, min(1.0, overall)), 4)

    def _generate_suggestions(self, result: RAGEvaluationResult) -> List[str]:
        """生成改进建议"""
        suggestions = []

        # 检索质量建议
        if result.retrieval.num_contexts == 0:
            suggestions.append("未检索到任何上下文。建议检查知识库索引或调整检索参数。")
        elif result.retrieval.context_precision < 0.5 and result.retrieval.context_precision > 0:
            suggestions.append(
                f"上下文精确度较低 ({result.retrieval.context_precision:.2f})。"
                "建议优化重排序模型或调整检索阈值。"
            )

        if result.retrieval.context_recall < 0.5 and result.retrieval.context_recall > 0:
            suggestions.append(
                f"上下文召回率不足 ({result.retrieval.context_recall:.2f})。"
                "建议增加检索数量或使用查询扩展。"
            )

        # 生成质量建议
        if result.generation.faithfulness < 0.7 and result.generation.faithfulness > 0:
            suggestions.append(
                f"答案忠实度偏低 ({result.generation.faithfulness:.2f})。"
                "答案可能包含上下文中不存在的信息。建议调整生成温度或优化提示词。"
            )

        relevancy = max(result.generation.relevancy, result.generation.answer_relevancy)
        if relevancy < 0.6 and relevancy > 0:
            suggestions.append(
                f"答案相关性不足 ({relevancy:.2f})。"
                "答案可能未直接回答问题。建议优化 RAG 提示词模板。"
            )

        if result.generation.correctness < 0.5 and result.generation.correctness > 0:
            suggestions.append(
                f"答案正确性较低 ({result.generation.correctness:.2f})。"
                "答案与参考答案差异较大。建议检查知识库内容质量。"
            )

        return suggestions

    def _log_result(self, result: RAGEvaluationResult) -> None:
        """记录评估结果"""
        quality = "excellent" if result.overall_score >= 0.8 else (
            "good" if result.overall_score >= 0.6 else (
                "fair" if result.overall_score >= 0.4 else "poor"
            )
        )

        logger.info(
            f"[RAGEvaluator] 评估完成 ({result.evaluator_type}) - "
            f"质量: {quality}, "
            f"综合分: {result.overall_score:.2f}, "
            f"忠实度: {result.generation.faithfulness:.2f}, "
            f"相关性: {max(result.generation.relevancy, result.generation.answer_relevancy):.2f}, "
            f"耗时: {result.evaluation_time_ms}ms"
        )

    async def batch_evaluate(
        self,
        test_cases: List[Dict[str, Any]],
        max_concurrent: int = 3,
    ) -> List[RAGEvaluationResult]:
        """
        批量评估多个测试用例。

        Args:
            test_cases: 测试用例列表，每个包含 query, response, contexts, reference(可选)
            max_concurrent: 最大并发数

        Returns:
            评估结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def evaluate_with_limit(case: Dict[str, Any]) -> RAGEvaluationResult:
            async with semaphore:
                return await self.evaluate(
                    query=case.get("query", ""),
                    response=case.get("response", ""),
                    contexts=case.get("contexts", []),
                    reference=case.get("reference"),
                )

        tasks = [evaluate_with_limit(case) for case in test_cases]
        return await asyncio.gather(*tasks)

    def get_evaluation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取评估历史"""
        return [r.to_dict() for r in self._evaluation_history[-limit:]]

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """获取聚合指标"""
        if not self._evaluation_history:
            return {}

        def avg(values: List[float]) -> float:
            valid = [v for v in values if v > 0]
            return sum(valid) / len(valid) if valid else 0.0

        results = self._evaluation_history

        return {
            "num_evaluations": len(results),
            "avg_overall_score": round(avg([r.overall_score for r in results]), 4),
            "retrieval": {
                "avg_context_precision": round(avg([r.retrieval.context_precision for r in results]), 4),
                "avg_context_recall": round(avg([r.retrieval.context_recall for r in results]), 4),
            },
            "generation": {
                "avg_faithfulness": round(avg([r.generation.faithfulness for r in results]), 4),
                "avg_relevancy": round(avg([
                    max(r.generation.relevancy, r.generation.answer_relevancy) for r in results
                ]), 4),
                "avg_correctness": round(avg([r.generation.correctness for r in results if r.reference]), 4),
            },
            "evaluator_distribution": {
                "llamaindex": sum(1 for r in results if r.evaluator_type == "llamaindex"),
                "ragas": sum(1 for r in results if r.evaluator_type == "ragas"),
                "hybrid": sum(1 for r in results if r.evaluator_type == "hybrid"),
            },
        }

    def reset(self) -> None:
        """重置评估历史"""
        self._evaluation_history.clear()


# ============== Dataset Generator ==============

class RAGDatasetGenerator:
    """
    RAG 测试数据集生成器。

    基于 LlamaIndex DatasetGenerator，用于:
    - 从文档自动生成问答对
    - 创建评估测试集
    """

    def __init__(self, llm: Optional[LLM] = None):
        from llama_index.core.evaluation import DatasetGenerator
        self._llm = llm or LlamaSettings.llm
        self.generator_class = DatasetGenerator
        logger.info("[RAGDatasetGenerator] Initialized")

    async def generate_from_nodes(
        self,
        nodes: List[TextNode],
        num_questions_per_chunk: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        从节点生成问答对。

        Args:
            nodes: 文本节点列表
            num_questions_per_chunk: 每个块生成的问题数

        Returns:
            问答对列表
        """
        try:
            generator = self.generator_class.from_documents(
                nodes,
                llm=self._llm,
                num_questions_per_chunk=num_questions_per_chunk,
            )

            # 生成问题
            questions = await asyncio.to_thread(generator.generate_questions_from_nodes)

            return [
                {"question": q, "source_node": nodes[i % len(nodes)].text[:200]}
                for i, q in enumerate(questions)
            ]

        except Exception as e:
            logger.error(f"[RAGDatasetGenerator] Generation failed: {e}")
            return []

    def generate_from_documents_sync(
        self,
        documents: List[Any],
        num_questions_per_chunk: int = 2,
    ) -> List[str]:
        """同步版本的数据集生成"""
        try:
            generator = self.generator_class.from_documents(
                documents,
                llm=self._llm,
                num_questions_per_chunk=num_questions_per_chunk,
            )
            return generator.generate_questions_from_nodes()
        except Exception as e:
            logger.error(f"[RAGDatasetGenerator] Sync generation failed: {e}")
            return []


# ============== Pipeline Quality Monitor ==============

@dataclass
class PipelineStageMetrics:
    """RAG 流水线各阶段质量指标"""
    # 查询转换阶段
    query_expansion_count: int = 0
    hyde_generated: bool = False

    # 检索阶段
    retrieved_count: int = 0
    avg_retrieval_score: float = 0.0
    min_retrieval_score: float = 0.0
    max_retrieval_score: float = 0.0

    # 重排序阶段
    reranked_count: int = 0
    filtered_count: int = 0
    avg_rerank_score: float = 0.0

    # 后处理阶段
    dedup_removed: int = 0
    final_count: int = 0

    # 生成阶段
    context_length: int = 0
    answer_length: int = 0

    # 整体质量
    quality_grade: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_transform": {
                "expansion_count": self.query_expansion_count,
                "hyde_generated": self.hyde_generated,
            },
            "retrieval": {
                "count": self.retrieved_count,
                "avg_score": round(self.avg_retrieval_score, 4),
                "score_range": [round(self.min_retrieval_score, 4), round(self.max_retrieval_score, 4)],
            },
            "rerank": {
                "input_count": self.retrieved_count,
                "output_count": self.reranked_count,
                "filtered": self.filtered_count,
                "avg_score": round(self.avg_rerank_score, 4),
            },
            "postprocess": {
                "dedup_removed": self.dedup_removed,
                "final_count": self.final_count,
            },
            "generation": {
                "context_length": self.context_length,
                "answer_length": self.answer_length,
            },
            "quality_grade": self.quality_grade,
        }


class PipelineQualityMonitor:
    """RAG 流水线质量监控器（轻量级，不依赖 LLM）"""

    def __init__(self):
        self._stage_history: List[PipelineStageMetrics] = []

    def evaluate_pipeline_stages(
        self,
        queries: List[str],
        retrieved_nodes: List[Any],
        reranked_nodes: List[Any],
        final_nodes: List[Any],
        answer: str,
        hyde_used: bool = False,
    ) -> PipelineStageMetrics:
        """评估流水线各阶段质量"""
        metrics = PipelineStageMetrics()

        metrics.query_expansion_count = len(queries)
        metrics.hyde_generated = hyde_used

        metrics.retrieved_count = len(retrieved_nodes)
        if retrieved_nodes:
            scores = [getattr(n, 'score', 0) or 0 for n in retrieved_nodes]
            metrics.avg_retrieval_score = sum(scores) / len(scores) if scores else 0
            metrics.min_retrieval_score = min(scores) if scores else 0
            metrics.max_retrieval_score = max(scores) if scores else 0

        metrics.reranked_count = len(reranked_nodes)
        metrics.filtered_count = len(retrieved_nodes) - len(reranked_nodes)
        if reranked_nodes:
            rerank_scores = [getattr(n, 'score', 0) or 0 for n in reranked_nodes]
            metrics.avg_rerank_score = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0

        metrics.dedup_removed = len(reranked_nodes) - len(final_nodes)
        metrics.final_count = len(final_nodes)

        if final_nodes:
            context_text = " ".join([
                getattr(n.node, 'text', '') if hasattr(n, 'node') else str(n)
                for n in final_nodes
            ])
            metrics.context_length = len(context_text)
        metrics.answer_length = len(answer)

        metrics.quality_grade = self._calculate_grade(metrics)
        self._stage_history.append(metrics)

        return metrics

    def _calculate_grade(self, m: PipelineStageMetrics) -> str:
        score = 0.0
        if m.retrieved_count > 0:
            score += 0.2 * min(1.0, m.retrieved_count / 10)
            score += 0.2 * m.avg_retrieval_score
        if m.reranked_count > 0:
            score += 0.15 * min(1.0, m.reranked_count / 5)
            score += 0.15 * m.avg_rerank_score
        if m.final_count > 0:
            score += 0.15 * min(1.0, m.final_count / 5)
            score += 0.15 * (1.0 if m.answer_length > 50 else m.answer_length / 50)

        if score >= 0.8: return "excellent"
        elif score >= 0.6: return "good"
        elif score >= 0.4: return "fair"
        return "poor"

    def get_stage_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._stage_history[-limit:]]

    def get_stage_stats(self) -> Dict[str, Any]:
        if not self._stage_history:
            return {}

        def avg(values): return sum(values) / len(values) if values else 0.0
        h = self._stage_history

        return {
            "num_queries": len(h),
            "avg_retrieved": round(avg([m.retrieved_count for m in h]), 2),
            "avg_reranked": round(avg([m.reranked_count for m in h]), 2),
            "avg_final": round(avg([m.final_count for m in h]), 2),
            "avg_retrieval_score": round(avg([m.avg_retrieval_score for m in h]), 4),
            "avg_rerank_score": round(avg([m.avg_rerank_score for m in h]), 4),
            "quality_distribution": {
                grade: sum(1 for m in h if m.quality_grade == grade)
                for grade in ["excellent", "good", "fair", "poor"]
            },
        }

    def reset(self) -> None:
        self._stage_history.clear()


# ============== Global Instances ==============

_rag_evaluator: Optional[RAGEvaluator] = None
_pipeline_monitor: Optional[PipelineQualityMonitor] = None
_dataset_generator: Optional[RAGDatasetGenerator] = None


def get_rag_evaluator() -> RAGEvaluator:
    """获取全局 RAG 评估器"""
    global _rag_evaluator
    if _rag_evaluator is None:
        _rag_evaluator = RAGEvaluator()
    return _rag_evaluator


def get_ragas_evaluator() -> RAGEvaluator:
    """获取全局 RAGAS 评估器（别名）"""
    return get_rag_evaluator()


def get_pipeline_monitor() -> PipelineQualityMonitor:
    """获取全局流水线质量监控器"""
    global _pipeline_monitor
    if _pipeline_monitor is None:
        _pipeline_monitor = PipelineQualityMonitor()
    return _pipeline_monitor


def get_dataset_generator() -> RAGDatasetGenerator:
    """获取全局数据集生成器"""
    global _dataset_generator
    if _dataset_generator is None:
        _dataset_generator = RAGDatasetGenerator()
    return _dataset_generator


# ============== Convenience Functions ==============

async def evaluate_rag_quality(
    query: str,
    response: str,
    contexts: List[str],
    reference: Optional[str] = None,
) -> RAGEvaluationResult:
    """便捷函数：评估 RAG 质量"""
    evaluator = get_rag_evaluator()
    return await evaluator.evaluate(
        query=query,
        response=response,
        contexts=contexts,
        reference=reference,
    )
