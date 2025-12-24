"""
Enhanced Logging Utilities / 增强日志工具
==========================================

提供统一的结构化日志输出，用于 RAG、Agent、Memory 等模块的流程追踪。

功能特性
--------
1. **阶段标记**: 清晰的视觉分隔符标识处理阶段
2. **计时统计**: 自动记录各步骤耗时
3. **指标汇总**: 文档数量、分数分布等关键指标
4. **上下文跟踪**: 请求 ID、会话 ID 等上下文信息

使用示例
--------
```python
from app.utils.log_utils import LogContext, log_phase_start, log_phase_end, log_step

# 创建日志上下文
ctx = LogContext(module="RAG", request_id="req_123")

# 记录阶段开始
log_phase_start(ctx, "Query Processing")

# 记录步骤
log_step(ctx, "Query Transform", "Transforming query with HyDE...")

# 记录阶段结束（带统计）
log_phase_end(ctx, "Query Processing", metrics={"queries": 3, "elapsed_ms": 150})
```

Author: Intelligent Customer Service Team
Version: 1.0.0
"""
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
import json

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 日志格式常量
# ═══════════════════════════════════════════════════════════════════════════════

# 分隔符样式
SEPARATOR_HEAVY = "═" * 60
SEPARATOR_LIGHT = "─" * 60
SEPARATOR_DOT = "·" * 60

# 阶段标记符号
PHASE_START = "▶"
PHASE_END = "◀"
STEP_MARKER = "→"
SUCCESS_MARKER = "✓"
FAIL_MARKER = "✗"
INFO_MARKER = "ℹ"
WARN_MARKER = "⚠"
DEBUG_MARKER = "○"


# ═══════════════════════════════════════════════════════════════════════════════
# 日志上下文
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LogContext:
    """日志上下文，携带请求级别的追踪信息。"""
    module: str  # 模块名称，如 "RAG", "Agent", "Memory"
    request_id: Optional[str] = None  # 请求 ID
    session_id: Optional[str] = None  # 会话 ID
    user_id: Optional[str] = None  # 用户 ID
    start_time: float = field(default_factory=time.time)
    step_times: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_prefix(self) -> str:
        """获取日志前缀。"""
        parts = [f"[{self.module}]"]
        if self.request_id:
            parts.append(f"[{self.request_id[:8]}]")
        return " ".join(parts)

    def elapsed_ms(self) -> int:
        """获取从开始到现在的毫秒数。"""
        return int((time.time() - self.start_time) * 1000)

    def record_step_time(self, step_name: str, elapsed_ms: float):
        """记录步骤耗时。"""
        self.step_times[step_name] = elapsed_ms


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段日志函数
# ═══════════════════════════════════════════════════════════════════════════════

def log_phase_start(
    ctx: LogContext,
    phase_name: str,
    description: Optional[str] = None,
    input_summary: Optional[str] = None,
):
    """
    记录阶段开始。

    Args:
        ctx: 日志上下文
        phase_name: 阶段名称
        description: 阶段描述
        input_summary: 输入摘要
    """
    prefix = ctx.get_prefix()
    logger.info(f"{prefix} {SEPARATOR_HEAVY}")
    logger.info(f"{prefix} {PHASE_START} {phase_name.upper()} 开始")
    if description:
        logger.info(f"{prefix}   {description}")
    if input_summary:
        logger.info(f"{prefix}   输入: {input_summary[:100]}{'...' if len(input_summary) > 100 else ''}")
    logger.info(f"{prefix} {SEPARATOR_LIGHT}")


def log_phase_end(
    ctx: LogContext,
    phase_name: str,
    success: bool = True,
    metrics: Optional[Dict[str, Any]] = None,
    output_summary: Optional[str] = None,
    elapsed_ms: Optional[float] = None,
):
    """
    记录阶段结束。

    Args:
        ctx: 日志上下文
        phase_name: 阶段名称
        success: 是否成功
        metrics: 统计指标
        output_summary: 输出摘要
        elapsed_ms: 耗时（毫秒）
    """
    prefix = ctx.get_prefix()
    status = SUCCESS_MARKER if success else FAIL_MARKER
    elapsed = elapsed_ms or ctx.elapsed_ms()

    logger.info(f"{prefix} {SEPARATOR_LIGHT}")
    logger.info(f"{prefix} {PHASE_END} {phase_name.upper()} 完成 {status} ({elapsed:.0f}ms)")

    if metrics:
        metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        logger.info(f"{prefix}   统计: {metrics_str}")

    if output_summary:
        logger.info(f"{prefix}   输出: {output_summary[:100]}{'...' if len(output_summary) > 100 else ''}")

    logger.info(f"{prefix} {SEPARATOR_HEAVY}")


def log_step(
    ctx: LogContext,
    step_name: str,
    message: str,
    level: str = "info",
    data: Optional[Dict[str, Any]] = None,
):
    """
    记录处理步骤。

    Args:
        ctx: 日志上下文
        step_name: 步骤名称
        message: 日志消息
        level: 日志级别 (debug, info, warning, error)
        data: 附加数据
    """
    prefix = ctx.get_prefix()
    full_message = f"{prefix} {STEP_MARKER} [{step_name}] {message}"

    if data:
        data_str = json.dumps(data, ensure_ascii=False, default=str)
        if len(data_str) > 200:
            data_str = data_str[:200] + "..."
        full_message += f" | {data_str}"

    log_func = getattr(logger, level, logger.info)
    log_func(full_message)


def log_substep(
    ctx: LogContext,
    step_name: str,
    substep: str,
    message: str,
):
    """记录子步骤。"""
    prefix = ctx.get_prefix()
    logger.info(f"{prefix}   {DEBUG_MARKER} [{step_name}:{substep}] {message}")


# ═══════════════════════════════════════════════════════════════════════════════
# 专用日志函数
# ═══════════════════════════════════════════════════════════════════════════════

def log_query_info(ctx: LogContext, query: str, query_type: str = "original"):
    """记录查询信息。"""
    prefix = ctx.get_prefix()
    truncated = query[:80] + "..." if len(query) > 80 else query
    logger.info(f"{prefix} {INFO_MARKER} 查询 ({query_type}): \"{truncated}\"")


def log_documents_retrieved(
    ctx: LogContext,
    count: int,
    source: str,
    top_scores: Optional[List[float]] = None,
):
    """记录检索到的文档信息。"""
    prefix = ctx.get_prefix()
    msg = f"{prefix} {INFO_MARKER} 检索结果 [{source}]: {count} 个文档"
    if top_scores:
        scores_str = ", ".join([f"{s:.3f}" for s in top_scores[:3]])
        msg += f" | Top分数: [{scores_str}]"
    logger.info(msg)


def log_llm_call(
    ctx: LogContext,
    operation: str,
    model: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    elapsed_ms: Optional[float] = None,
):
    """记录 LLM 调用信息。"""
    prefix = ctx.get_prefix()
    parts = [f"{prefix} {INFO_MARKER} LLM调用 [{operation}]"]

    if model:
        parts.append(f"模型: {model}")
    if input_tokens:
        parts.append(f"输入: {input_tokens} tokens")
    if output_tokens:
        parts.append(f"输出: {output_tokens} tokens")
    if elapsed_ms:
        parts.append(f"耗时: {elapsed_ms:.0f}ms")

    logger.info(" | ".join(parts))


def log_tool_call(
    ctx: LogContext,
    tool_name: str,
    status: str,  # "start", "success", "error"
    params: Optional[Dict[str, Any]] = None,
    result_summary: Optional[str] = None,
    error: Optional[str] = None,
    elapsed_ms: Optional[float] = None,
):
    """记录工具调用信息。"""
    prefix = ctx.get_prefix()

    if status == "start":
        params_str = json.dumps(params, ensure_ascii=False)[:100] if params else ""
        logger.info(f"{prefix} {STEP_MARKER} 工具调用 [{tool_name}] 开始 | 参数: {params_str}")
    elif status == "success":
        msg = f"{prefix} {SUCCESS_MARKER} 工具调用 [{tool_name}] 成功"
        if elapsed_ms:
            msg += f" ({elapsed_ms:.0f}ms)"
        if result_summary:
            msg += f" | 结果: {result_summary[:100]}"
        logger.info(msg)
    elif status == "error":
        logger.error(f"{prefix} {FAIL_MARKER} 工具调用 [{tool_name}] 失败 | 错误: {error}")


def log_memory_operation(
    ctx: LogContext,
    operation: str,
    conversation_id: str,
    message_count: Optional[int] = None,
    has_summary: bool = False,
):
    """记录记忆操作。"""
    prefix = ctx.get_prefix()
    msg = f"{prefix} {INFO_MARKER} 记忆操作 [{operation}] 会话: {conversation_id[:8]}"
    if message_count is not None:
        msg += f" | 消息数: {message_count}"
    if has_summary:
        msg += " | 已摘要"
    logger.info(msg)


def log_scores_distribution(
    ctx: LogContext,
    step_name: str,
    scores: List[float],
):
    """记录分数分布统计。"""
    if not scores:
        return

    prefix = ctx.get_prefix()
    import statistics

    stats = {
        "min": min(scores),
        "max": max(scores),
        "avg": statistics.mean(scores),
        "count": len(scores),
    }
    if len(scores) > 1:
        stats["std"] = statistics.stdev(scores)

    stats_str = " | ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()])
    logger.info(f"{prefix}   分数统计 [{step_name}]: {stats_str}")


def log_timing_summary(ctx: LogContext):
    """记录耗时汇总。"""
    if not ctx.step_times:
        return

    prefix = ctx.get_prefix()
    total = ctx.elapsed_ms()

    logger.info(f"{prefix} {SEPARATOR_DOT}")
    logger.info(f"{prefix} 耗时汇总 (总计: {total}ms)")

    for step, elapsed in ctx.step_times.items():
        percentage = (elapsed / total * 100) if total > 0 else 0
        bar_len = int(percentage / 5)  # 每 5% 一个字符
        bar = "█" * bar_len + "░" * (20 - bar_len)
        logger.info(f"{prefix}   {step:20s} {elapsed:6.0f}ms ({percentage:5.1f}%) {bar}")


# ═══════════════════════════════════════════════════════════════════════════════
# 上下文管理器和装饰器
# ═══════════════════════════════════════════════════════════════════════════════

@contextmanager
def log_phase(
    ctx: LogContext,
    phase_name: str,
    description: Optional[str] = None,
):
    """
    阶段日志上下文管理器。

    Usage:
        with log_phase(ctx, "Retrieval") as phase:
            # do retrieval
            phase.metrics["doc_count"] = 10
    """
    class PhaseContext:
        def __init__(self):
            self.start_time = time.time()
            self.metrics: Dict[str, Any] = {}
            self.success = True
            self.output_summary = None

    phase = PhaseContext()
    log_phase_start(ctx, phase_name, description)

    try:
        yield phase
    except Exception as e:
        phase.success = False
        phase.metrics["error"] = str(e)
        raise
    finally:
        elapsed = (time.time() - phase.start_time) * 1000
        ctx.record_step_time(phase_name, elapsed)
        log_phase_end(
            ctx,
            phase_name,
            success=phase.success,
            metrics=phase.metrics,
            output_summary=phase.output_summary,
            elapsed_ms=elapsed,
        )


@contextmanager
def log_step_timer(ctx: LogContext, step_name: str):
    """
    步骤计时上下文管理器。

    Usage:
        with log_step_timer(ctx, "embedding"):
            # compute embeddings
    """
    start = time.time()
    log_step(ctx, step_name, "开始处理...")

    try:
        yield
    finally:
        elapsed = (time.time() - start) * 1000
        ctx.record_step_time(step_name, elapsed)
        log_step(ctx, step_name, f"处理完成 ({elapsed:.0f}ms)")


def timed_operation(step_name: str):
    """
    计时装饰器，用于函数级别的耗时记录。

    Usage:
        @timed_operation("generate_response")
        async def generate_response(self, query):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"[Timing] {step_name} 开始")
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.time() - start) * 1000
                logger.info(f"[Timing] {step_name} 完成 ({elapsed:.0f}ms)")
                return result
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                logger.error(f"[Timing] {step_name} 失败 ({elapsed:.0f}ms): {e}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"[Timing] {step_name} 开始")
            try:
                result = func(*args, **kwargs)
                elapsed = (time.time() - start) * 1000
                logger.info(f"[Timing] {step_name} 完成 ({elapsed:.0f}ms)")
                return result
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                logger.error(f"[Timing] {step_name} 失败 ({elapsed:.0f}ms): {e}")
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# 专用模块日志类
# ═══════════════════════════════════════════════════════════════════════════════

class RAGLogger:
    """RAG 流程专用日志记录器。"""

    def __init__(self, request_id: Optional[str] = None):
        self.ctx = LogContext(module="RAG", request_id=request_id)

    def start_query(self, query: str, user_id: Optional[str] = None):
        """记录查询开始。"""
        self.ctx.user_id = user_id
        self.ctx.start_time = time.time()
        log_phase_start(self.ctx, "RAG Query", f"处理用户查询")
        log_query_info(self.ctx, query, "original")

    def log_transform(self, original: str, transformed: List[str], elapsed_ms: float):
        """记录查询转换。"""
        log_step(self.ctx, "QueryTransform", f"生成 {len(transformed)} 个查询变体 ({elapsed_ms:.0f}ms)")
        for i, q in enumerate(transformed[:3]):
            log_substep(self.ctx, "QueryTransform", f"Q{i+1}", q[:60])
        self.ctx.record_step_time("QueryTransform", elapsed_ms)

    def log_retrieval(self, source: str, count: int, scores: List[float], elapsed_ms: float):
        """记录检索结果。"""
        log_documents_retrieved(self.ctx, count, source, scores[:5])
        log_scores_distribution(self.ctx, source, scores)
        self.ctx.record_step_time(f"Retrieval_{source}", elapsed_ms)

    def log_rerank(self, before: int, after: int, scores: List[float], elapsed_ms: float):
        """记录重排序结果。"""
        log_step(self.ctx, "Rerank", f"重排序 {before} → {after} 文档 ({elapsed_ms:.0f}ms)")
        log_scores_distribution(self.ctx, "Rerank", scores)
        self.ctx.record_step_time("Rerank", elapsed_ms)

    def log_postprocess(self, before: int, after: int, elapsed_ms: float):
        """记录后处理结果。"""
        log_step(self.ctx, "PostProcess", f"后处理 {before} → {after} 文档 ({elapsed_ms:.0f}ms)")
        self.ctx.record_step_time("PostProcess", elapsed_ms)

    def log_generation(self, context_length: int, response_length: int, elapsed_ms: float):
        """记录响应生成。"""
        log_llm_call(
            self.ctx, "ResponseGen",
            input_tokens=context_length // 4,
            output_tokens=response_length // 4,
            elapsed_ms=elapsed_ms
        )
        self.ctx.record_step_time("Generation", elapsed_ms)

    def end_query(self, success: bool, answer_length: int, source_count: int):
        """记录查询结束。"""
        log_timing_summary(self.ctx)
        log_phase_end(
            self.ctx, "RAG Query",
            success=success,
            metrics={
                "answer_length": answer_length,
                "sources": source_count,
                "total_ms": self.ctx.elapsed_ms()
            }
        )


class AgentLogger:
    """Agent 流程专用日志记录器。"""

    def __init__(self, request_id: Optional[str] = None):
        self.ctx = LogContext(module="Agent", request_id=request_id)
        self.iteration = 0

    def start_run(self, question: str, max_iterations: int):
        """记录 Agent 运行开始。"""
        self.ctx.start_time = time.time()
        log_phase_start(self.ctx, "Agent Run", f"最大迭代: {max_iterations}")
        log_query_info(self.ctx, question, "user_question")

    def log_planning(self, task_count: int, elapsed_ms: float):
        """记录任务规划。"""
        log_step(self.ctx, "Planning", f"生成 {task_count} 个任务步骤 ({elapsed_ms:.0f}ms)")
        self.ctx.record_step_time("Planning", elapsed_ms)

    def log_thinking(self, iteration: int, thought: str, action: Optional[str]):
        """记录思考过程。"""
        self.iteration = iteration
        prefix = self.ctx.get_prefix()
        logger.info(f"{prefix} {SEPARATOR_LIGHT}")
        logger.info(f"{prefix} 迭代 #{iteration}")
        logger.info(f"{prefix}   思考: {thought[:100]}{'...' if len(thought) > 100 else ''}")
        if action:
            logger.info(f"{prefix}   决策: 执行工具 [{action}]")
        else:
            logger.info(f"{prefix}   决策: 生成最终回答")

    def log_action(self, tool_name: str, params: Dict[str, Any]):
        """记录动作执行开始。"""
        log_tool_call(self.ctx, tool_name, "start", params)

    def log_observation(self, tool_name: str, success: bool, result: str, elapsed_ms: float):
        """记录观察结果。"""
        if success:
            log_tool_call(self.ctx, tool_name, "success", result_summary=result, elapsed_ms=elapsed_ms)
        else:
            log_tool_call(self.ctx, tool_name, "error", error=result, elapsed_ms=elapsed_ms)
        self.ctx.record_step_time(f"Tool_{tool_name}", elapsed_ms)

    def log_response(self, response_length: int, elapsed_ms: float):
        """记录最终响应生成。"""
        log_step(self.ctx, "Response", f"生成回答 ({response_length} 字符, {elapsed_ms:.0f}ms)")
        self.ctx.record_step_time("Response", elapsed_ms)

    def end_run(self, success: bool, iterations: int, tool_calls: int):
        """记录 Agent 运行结束。"""
        log_timing_summary(self.ctx)
        log_phase_end(
            self.ctx, "Agent Run",
            success=success,
            metrics={
                "iterations": iterations,
                "tool_calls": tool_calls,
                "total_ms": self.ctx.elapsed_ms()
            }
        )


class MemoryLogger:
    """记忆管理专用日志记录器。"""

    def __init__(self):
        self.ctx = LogContext(module="Memory")

    def log_add_message(self, conversation_id: str, role: str, msg_count: int):
        """记录添加消息。"""
        log_step(
            self.ctx, "AddMessage",
            f"会话 {conversation_id[:8]} | 角色: {role} | 总消息: {msg_count}"
        )

    def log_summarize_start(self, conversation_id: str, msg_count: int, keep_recent: int):
        """记录开始摘要。"""
        log_step(
            self.ctx, "Summarize",
            f"会话 {conversation_id[:8]} | 消息数 {msg_count} 触发摘要 (保留最近 {keep_recent} 条)"
        )

    def log_summarize_complete(self, conversation_id: str, summary_length: int, elapsed_ms: float):
        """记录摘要完成。"""
        log_step(
            self.ctx, "Summarize",
            f"会话 {conversation_id[:8]} | 摘要生成完成 ({summary_length} 字符, {elapsed_ms:.0f}ms)"
        )

    def log_get_context(self, conversation_id: str, msg_count: int, has_summary: bool):
        """记录获取上下文。"""
        log_memory_operation(self.ctx, "GetContext", conversation_id, msg_count, has_summary)


# ═══════════════════════════════════════════════════════════════════════════════
# 结果日志记录器
# ═══════════════════════════════════════════════════════════════════════════════

class ResultLogger:
    """结果日志记录器，用于记录最终处理结果。"""

    def __init__(self, module: str = "Result", request_id: Optional[str] = None):
        self.ctx = LogContext(module=module, request_id=request_id)

    def log_success(
        self,
        operation: str,
        result_summary: str,
        metrics: Optional[Dict[str, Any]] = None,
        elapsed_ms: Optional[float] = None,
    ):
        """记录成功结果。"""
        prefix = self.ctx.get_prefix()
        msg = f"{prefix} {SUCCESS_MARKER} {operation} 成功"
        if elapsed_ms:
            msg += f" ({elapsed_ms:.0f}ms)"
        logger.info(msg)

        if result_summary:
            logger.info(f"{prefix}   结果: {result_summary[:150]}{'...' if len(result_summary) > 150 else ''}")

        if metrics:
            metrics_str = " | ".join([f"{k}={v}" for k, v in metrics.items()])
            logger.info(f"{prefix}   指标: {metrics_str}")

    def log_failure(
        self,
        operation: str,
        error: str,
        error_type: Optional[str] = None,
        elapsed_ms: Optional[float] = None,
    ):
        """记录失败结果。"""
        prefix = self.ctx.get_prefix()
        msg = f"{prefix} {FAIL_MARKER} {operation} 失败"
        if elapsed_ms:
            msg += f" ({elapsed_ms:.0f}ms)"
        logger.error(msg)

        if error_type:
            logger.error(f"{prefix}   错误类型: {error_type}")
        logger.error(f"{prefix}   错误信息: {error[:200]}{'...' if len(error) > 200 else ''}")

    def log_partial(
        self,
        operation: str,
        message: str,
        success_count: int = 0,
        failure_count: int = 0,
    ):
        """记录部分成功结果。"""
        prefix = self.ctx.get_prefix()
        logger.warning(f"{prefix} {WARN_MARKER} {operation} 部分完成")
        logger.warning(f"{prefix}   {message}")
        logger.warning(f"{prefix}   成功: {success_count} | 失败: {failure_count}")


# ═══════════════════════════════════════════════════════════════════════════════
# 性能日志记录器
# ═══════════════════════════════════════════════════════════════════════════════

class PerformanceLogger:
    """性能日志记录器，用于详细的性能分析。"""

    def __init__(self, module: str = "Perf", request_id: Optional[str] = None):
        self.ctx = LogContext(module=module, request_id=request_id)
        self.checkpoints: List[tuple] = []  # (name, timestamp)
        self.metrics: Dict[str, Any] = {}

    def start(self, operation: str):
        """开始性能追踪。"""
        self.ctx.start_time = time.time()
        self.checkpoints = [("start", time.time())]
        prefix = self.ctx.get_prefix()
        logger.debug(f"{prefix} [Perf] 开始追踪: {operation}")

    def checkpoint(self, name: str):
        """记录检查点。"""
        self.checkpoints.append((name, time.time()))
        if len(self.checkpoints) > 1:
            prev_name, prev_time = self.checkpoints[-2]
            elapsed = (time.time() - prev_time) * 1000
            self.ctx.record_step_time(name, elapsed)

    def record_metric(self, name: str, value: Any):
        """记录性能指标。"""
        self.metrics[name] = value

    def end(self, operation: str):
        """结束性能追踪并输出汇总。"""
        self.checkpoints.append(("end", time.time()))
        total_elapsed = self.ctx.elapsed_ms()

        prefix = self.ctx.get_prefix()
        logger.info(f"{prefix} {SEPARATOR_DOT}")
        logger.info(f"{prefix} [Perf] {operation} 性能汇总 (总耗时: {total_elapsed}ms)")

        # 输出各阶段耗时
        if len(self.checkpoints) > 2:
            for i in range(1, len(self.checkpoints)):
                name, ts = self.checkpoints[i]
                prev_name, prev_ts = self.checkpoints[i-1]
                elapsed = (ts - prev_ts) * 1000
                percentage = (elapsed / total_elapsed * 100) if total_elapsed > 0 else 0
                bar_len = int(percentage / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                logger.info(f"{prefix}   {prev_name} → {name}: {elapsed:6.0f}ms ({percentage:5.1f}%) {bar}")

        # 输出其他指标
        if self.metrics:
            logger.info(f"{prefix}   ────────────────────")
            for name, value in self.metrics.items():
                logger.info(f"{prefix}   {name}: {value}")


# ═══════════════════════════════════════════════════════════════════════════════
# 配置日志函数
# ═══════════════════════════════════════════════════════════════════════════════

def log_config(
    ctx: LogContext,
    config_name: str,
    config_values: Dict[str, Any],
    sensitive_keys: Optional[List[str]] = None,
):
    """
    记录配置信息。

    Args:
        ctx: 日志上下文
        config_name: 配置名称
        config_values: 配置值字典
        sensitive_keys: 敏感字段列表（将被掩码）
    """
    prefix = ctx.get_prefix()
    sensitive_keys = sensitive_keys or []

    logger.info(f"{prefix} {INFO_MARKER} 配置 [{config_name}]")
    for key, value in config_values.items():
        if key.lower() in [k.lower() for k in sensitive_keys]:
            display_value = "***MASKED***"
        elif isinstance(value, str) and len(value) > 50:
            display_value = value[:50] + "..."
        else:
            display_value = value
        logger.info(f"{prefix}   {key}: {display_value}")


def log_request_start(
    ctx: LogContext,
    endpoint: str,
    method: str = "POST",
    params: Optional[Dict[str, Any]] = None,
):
    """记录请求开始。"""
    prefix = ctx.get_prefix()
    logger.info(f"{prefix} {SEPARATOR_HEAVY}")
    logger.info(f"{prefix} {PHASE_START} 请求开始: {method} {endpoint}")
    if params:
        params_str = json.dumps(params, ensure_ascii=False, default=str)
        if len(params_str) > 200:
            params_str = params_str[:200] + "..."
        logger.info(f"{prefix}   参数: {params_str}")


def log_request_end(
    ctx: LogContext,
    endpoint: str,
    status_code: int = 200,
    response_summary: Optional[str] = None,
):
    """记录请求结束。"""
    prefix = ctx.get_prefix()
    elapsed = ctx.elapsed_ms()

    status_marker = SUCCESS_MARKER if 200 <= status_code < 400 else FAIL_MARKER
    logger.info(f"{prefix} {PHASE_END} 请求完成 {status_marker} | 状态: {status_code} | 耗时: {elapsed}ms")

    if response_summary:
        logger.info(f"{prefix}   响应: {response_summary[:100]}{'...' if len(response_summary) > 100 else ''}")

    logger.info(f"{prefix} {SEPARATOR_HEAVY}")


# ═══════════════════════════════════════════════════════════════════════════════
# 错误日志增强
# ═══════════════════════════════════════════════════════════════════════════════

def log_error_detail(
    ctx: LogContext,
    error: Exception,
    operation: str,
    context_data: Optional[Dict[str, Any]] = None,
    include_traceback: bool = True,
):
    """
    记录详细的错误信息。

    Args:
        ctx: 日志上下文
        error: 异常对象
        operation: 操作名称
        context_data: 上下文数据
        include_traceback: 是否包含堆栈跟踪
    """
    import traceback

    prefix = ctx.get_prefix()
    error_type = type(error).__name__

    logger.error(f"{prefix} {FAIL_MARKER} 错误发生于 [{operation}]")
    logger.error(f"{prefix}   错误类型: {error_type}")
    logger.error(f"{prefix}   错误消息: {str(error)[:300]}")

    if context_data:
        logger.error(f"{prefix}   上下文数据:")
        for key, value in context_data.items():
            value_str = str(value)[:100]
            logger.error(f"{prefix}     {key}: {value_str}")

    if include_traceback:
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        logger.error(f"{prefix}   堆栈跟踪:")
        for line in tb_lines[-5:]:  # 只显示最后5行
            for subline in line.strip().split('\n'):
                logger.error(f"{prefix}     {subline}")


def log_warning_detail(
    ctx: LogContext,
    message: str,
    warning_type: str = "Warning",
    suggestion: Optional[str] = None,
):
    """记录详细的警告信息。"""
    prefix = ctx.get_prefix()

    logger.warning(f"{prefix} {WARN_MARKER} [{warning_type}] {message}")
    if suggestion:
        logger.warning(f"{prefix}   建议: {suggestion}")


# ═══════════════════════════════════════════════════════════════════════════════
# 流程追踪日志
# ═══════════════════════════════════════════════════════════════════════════════

def log_pipeline_start(
    ctx: LogContext,
    pipeline_name: str,
    stages: List[str],
):
    """记录流水线开始。"""
    prefix = ctx.get_prefix()
    logger.info(f"{prefix} {SEPARATOR_HEAVY}")
    logger.info(f"{prefix} {PHASE_START} 流水线 [{pipeline_name}] 开始")
    logger.info(f"{prefix}   阶段: {' → '.join(stages)}")
    logger.info(f"{prefix} {SEPARATOR_LIGHT}")


def log_pipeline_stage(
    ctx: LogContext,
    stage_name: str,
    stage_index: int,
    total_stages: int,
    status: str = "running",  # running, completed, skipped, failed
):
    """记录流水线阶段状态。"""
    prefix = ctx.get_prefix()

    status_markers = {
        "running": "▶",
        "completed": SUCCESS_MARKER,
        "skipped": "⊘",
        "failed": FAIL_MARKER,
    }
    marker = status_markers.get(status, "○")

    logger.info(f"{prefix} {marker} [{stage_index}/{total_stages}] {stage_name} - {status}")


def log_pipeline_end(
    ctx: LogContext,
    pipeline_name: str,
    success: bool = True,
    completed_stages: int = 0,
    total_stages: int = 0,
):
    """记录流水线结束。"""
    prefix = ctx.get_prefix()
    elapsed = ctx.elapsed_ms()

    status = SUCCESS_MARKER if success else FAIL_MARKER
    logger.info(f"{prefix} {SEPARATOR_LIGHT}")
    logger.info(f"{prefix} {PHASE_END} 流水线 [{pipeline_name}] 完成 {status}")
    logger.info(f"{prefix}   完成阶段: {completed_stages}/{total_stages} | 总耗时: {elapsed}ms")
    log_timing_summary(ctx)
    logger.info(f"{prefix} {SEPARATOR_HEAVY}")