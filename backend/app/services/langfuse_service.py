"""
Langfuse Service / 可观测性服务
================================

基于 Langfuse 的 RAG 流程监控和追踪服务。

核心功能
--------
1. **追踪管理**: 创建、管理和结束追踪会话
2. **Span 管理**: 支持嵌套的操作追踪
3. **指标记录**: 记录 LLM 调用、检索、嵌入等指标
4. **成本追踪**: 自动计算 Token 使用和成本

追踪层级
--------
```
Trace (用户查询)
├── Span: Query Transform
│   ├── Generation: HyDE
│   └── Generation: Query Expansion
├── Span: Retrieval
│   ├── Span: Vector Search
│   └── Span: BM25 Search
├── Span: Reranking
├── Span: Post-processing
└── Generation: Response
```

使用示例
--------
```python
from app.services.langfuse_service import LangfuseService

# 初始化
langfuse = LangfuseService()

# 创建追踪
trace = langfuse.create_trace(
    name="rag_query",
    user_id="user_123",
    input={"question": "如何申请退款？"}
)

# 创建 Span
with langfuse.span(trace, "retrieval") as span:
    results = retriever.retrieve(query)
    span.set_output({"num_results": len(results)})

# 记录 LLM 调用
langfuse.log_generation(
    trace=trace,
    name="response_generation",
    model="qwen2.5:7b",
    input=prompt,
    output=response,
    usage={"input_tokens": 100, "output_tokens": 200}
)

# 结束追踪
langfuse.end_trace(trace, output={"answer": response})
```

Author: Intelligent Customer Service Team
Version: 1.0.0
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import contextmanager
from functools import wraps

from config.settings import settings

logger = logging.getLogger(__name__)

# Try to import langfuse
LANGFUSE_AVAILABLE = False
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
    logger.debug("[Langfuse] Package imported successfully")
except ImportError as e:
    logger.warning(
        f"[Langfuse] langfuse package not installed. "
        f"Install with: pip install langfuse. Error: {e}"
    )
    Langfuse = None

# Optional decorator imports (may not be available in all versions)
observe = None
langfuse_context = None
try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    pass  # Decorators not available in this version


class LangfuseService:
    """
    Langfuse 可观测性服务
    ====================

    提供 RAG 流程的追踪、监控和分析功能。

    Attributes
    ----------
    enabled : bool
        是否启用 Langfuse

    client : Langfuse
        Langfuse 客户端实例

    Example
    -------
    ```python
    service = LangfuseService()

    # 创建追踪
    trace = service.create_trace("rag_query", user_id="user_1")

    # 记录操作
    service.log_span(trace, "retrieval", input_data, output_data)

    # 记录 LLM 调用
    service.log_generation(trace, "llm_response", ...)
    ```
    """

    def __init__(
        self,
        secret_key: str = None,
        public_key: str = None,
        host: str = None,
        enabled: bool = None,
    ):
        """
        初始化 Langfuse 服务。

        Parameters
        ----------
        secret_key : str, optional
            Langfuse Secret Key，默认使用 settings.LANGFUSE_SECRET_KEY

        public_key : str, optional
            Langfuse Public Key，默认使用 settings.LANGFUSE_PUBLIC_KEY

        host : str, optional
            Langfuse Host，默认使用 settings.LANGFUSE_HOST

        enabled : bool, optional
            是否启用，默认使用 settings.LANGFUSE_ENABLED
        """
        self.enabled = (
            enabled if enabled is not None else settings.LANGFUSE_ENABLED
        )

        if not LANGFUSE_AVAILABLE:
            self.enabled = False
            self._client = None
            logger.warning("[Langfuse] Service disabled - package not installed")
            return

        if not self.enabled:
            self._client = None
            logger.info("[Langfuse] Service disabled by configuration")
            return

        self._secret_key = secret_key or settings.LANGFUSE_SECRET_KEY
        self._public_key = public_key or settings.LANGFUSE_PUBLIC_KEY
        self._host = host or settings.LANGFUSE_HOST

        if not self._secret_key or not self._public_key:
            self.enabled = False
            self._client = None
            logger.warning(
                "[Langfuse] Service disabled - missing API keys. "
                "Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY in .env"
            )
            return

        try:
            self._client = Langfuse(
                secret_key=self._secret_key,
                public_key=self._public_key,
                host=self._host,
            )
            logger.info(f"[Langfuse] Initialized - host: {self._host}")
        except Exception as e:
            self.enabled = False
            self._client = None
            logger.error(f"[Langfuse] Initialization failed: {e}")

    @property
    def client(self) -> Optional["Langfuse"]:
        """获取 Langfuse 客户端实例。"""
        return self._client

    def create_trace(
        self,
        name: str,
        user_id: str = None,
        session_id: str = None,
        input: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        tags: List[str] = None,
    ):
        """
        创建新的追踪。

        Parameters
        ----------
        name : str
            追踪名称

        user_id : str, optional
            用户 ID

        session_id : str, optional
            会话 ID

        input : dict, optional
            输入数据

        metadata : dict, optional
            元数据

        tags : list, optional
            标签列表

        Returns
        -------
        Trace | None
            Langfuse LangfuseSpan 对象 (作为根追踪)
        """
        if not self.enabled or not self._client:
            return None

        try:
            # Langfuse SDK: 使用 start_span 创建根 span (自动创建 trace)
            trace = self._client.start_span(
                name=name,
                input=input,
                metadata=metadata or {},
            )

            # 更新 trace 级别信息
            if user_id or session_id or tags:
                trace.update_trace(
                    user_id=user_id,
                    session_id=session_id,
                    tags=tags or [],
                )

            logger.debug(f"[Langfuse] Created trace: {name}, trace_id: {trace.trace_id}")
            return trace
        except Exception as e:
            logger.error(f"[Langfuse] Failed to create trace: {e}")
            return None

    def end_trace(
        self,
        trace,
        output: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        结束追踪并更新输出。

        Parameters
        ----------
        trace : Trace
            Langfuse LangfuseSpan 对象

        output : dict, optional
            输出数据

        metadata : dict, optional
            额外元数据
        """
        if not trace:
            return

        try:
            # 先更新输出
            update_kwargs = {}
            if output is not None:
                update_kwargs["output"] = output
            if metadata is not None:
                update_kwargs["metadata"] = metadata

            if update_kwargs:
                trace.update(**update_kwargs)

            # 结束 span
            trace.end()
            logger.debug(f"[Langfuse] Ended trace")
        except Exception as e:
            logger.error(f"[Langfuse] Failed to end trace: {e}")

    def create_span(
        self,
        trace,
        name: str,
        input: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        创建追踪 Span。

        Parameters
        ----------
        trace : Trace
            父 Trace/Span 对象

        name : str
            Span 名称

        input : dict, optional
            输入数据

        metadata : dict, optional
            元数据

        Returns
        -------
        Span | None
            Langfuse Span 对象
        """
        if not trace or not self._client:
            return None

        try:
            # 使用父对象的 start_span() 方法创建子 span
            span = trace.start_span(
                name=name,
                input=input,
                metadata=metadata or {},
            )
            logger.debug(f"[Langfuse] Created span: {name}")
            return span
        except Exception as e:
            logger.error(f"[Langfuse] Failed to create span: {e}")
            return None

    def end_span(
        self,
        span,
        output: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        level: str = None,
        status_message: str = None,
    ):
        """
        结束 Span。

        Parameters
        ----------
        span : Span
            Span 对象

        output : dict, optional
            输出数据

        metadata : dict, optional
            额外元数据

        level : str, optional
            日志级别 (DEBUG, INFO, WARNING, ERROR)

        status_message : str, optional
            状态消息
        """
        if not span:
            return

        try:
            # Langfuse v3: 先 update 设置输出，再 end
            update_kwargs = {}
            if output is not None:
                update_kwargs["output"] = output
            if metadata is not None:
                update_kwargs["metadata"] = metadata
            if level is not None:
                update_kwargs["level"] = level
            if status_message is not None:
                update_kwargs["status_message"] = status_message

            if update_kwargs:
                span.update(**update_kwargs)
            span.end()
            logger.debug(f"[Langfuse] Ended span")
        except Exception as e:
            logger.error(f"[Langfuse] Failed to end span: {e}")

    @contextmanager
    def span(
        self,
        trace,
        name: str,
        input: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        Span 上下文管理器。

        Example
        -------
        ```python
        with langfuse.span(trace, "retrieval") as span:
            results = retriever.retrieve(query)
            span.output = {"num_results": len(results)}
        ```
        """
        span = self.create_span(trace, name, input, metadata)

        class SpanContext:
            def __init__(self, span_obj):
                self._span = span_obj
                self.output = None
                self.metadata = None
                self.level = None
                self.status_message = None

            def set_output(self, output):
                self.output = output

        ctx = SpanContext(span)
        try:
            yield ctx
        except Exception as e:
            ctx.level = "ERROR"
            ctx.status_message = str(e)
            raise
        finally:
            self.end_span(
                span,
                output=ctx.output,
                metadata=ctx.metadata,
                level=ctx.level,
                status_message=ctx.status_message,
            )

    def log_generation(
        self,
        trace,
        name: str,
        model: str,
        input: Any = None,
        output: Any = None,
        usage: Dict[str, int] = None,
        metadata: Dict[str, Any] = None,
        parent_span=None,
    ):
        """
        记录 LLM 调用。

        Parameters
        ----------
        trace : Trace
            父 Trace 对象

        name : str
            生成名称

        model : str
            模型名称

        input : Any, optional
            输入（prompt）

        output : Any, optional
            输出（response）

        usage : dict, optional
            Token 使用量，如 {"input_tokens": 100, "output_tokens": 200}

        metadata : dict, optional
            元数据

        parent_span : Span, optional
            父 Span

        Returns
        -------
        Generation | None
            Generation 对象
        """
        if not trace or not self._client:
            return None

        try:
            parent = parent_span or trace

            # 使用父对象的 start_generation() 方法
            generation = parent.start_generation(
                name=name,
                model=model,
                input=input,
                metadata=metadata or {},
            )
            # 更新输出并结束
            if output is not None or usage is not None:
                update_kwargs = {}
                if output is not None:
                    update_kwargs["output"] = output
                if usage is not None:
                    update_kwargs["usage"] = usage
                generation.update(**update_kwargs)
            generation.end()
            logger.debug(f"[Langfuse] Logged generation: {name}, model: {model}")
            return generation
        except Exception as e:
            logger.error(f"[Langfuse] Failed to log generation: {e}")
            return None

    def log_retrieval(
        self,
        trace,
        name: str,
        query: str,
        documents: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None,
        parent_span=None,
    ):
        """
        记录检索操作。

        Parameters
        ----------
        trace : Trace
            父 Trace 对象

        name : str
            检索名称

        query : str
            查询文本

        documents : list
            检索结果列表

        metadata : dict, optional
            元数据

        parent_span : Span, optional
            父 Span
        """
        if not trace or not self._client:
            return None

        try:
            parent = parent_span or trace

            # 使用父对象的 start_span() 方法
            span = parent.start_span(
                name=name,
                input={"query": query},
                metadata={
                    "type": "retrieval",
                    **(metadata or {}),
                },
            )
            # 更新输出并结束
            span.update(
                output={
                    "num_documents": len(documents),
                    "documents": [
                        {
                            "content": doc.get("content", "")[:200],
                            "score": doc.get("score"),
                            "source": doc.get("source"),
                        }
                        for doc in documents[:5]  # 限制数量
                    ],
                },
            )
            span.end()
            logger.debug(f"[Langfuse] Logged retrieval: {name}, docs: {len(documents)}")
            return span
        except Exception as e:
            logger.error(f"[Langfuse] Failed to log retrieval: {e}")
            return None

    def log_embedding(
        self,
        trace,
        name: str,
        model: str,
        texts: List[str],
        metadata: Dict[str, Any] = None,
        parent_span=None,
    ):
        """
        记录嵌入操作。

        Parameters
        ----------
        trace : Trace
            父 Trace 对象

        name : str
            嵌入操作名称

        model : str
            嵌入模型名称

        texts : list
            嵌入的文本列表

        metadata : dict, optional
            元数据

        parent_span : Span, optional
            父 Span
        """
        if not trace or not self._client:
            return None

        try:
            parent = parent_span or trace

            # 使用父对象的 start_span() 方法
            span = parent.start_span(
                name=name,
                input={"num_texts": len(texts)},
                metadata={
                    "type": "embedding",
                    "model": model,
                    **(metadata or {}),
                },
            )
            # 更新输出并结束
            span.update(output={"model": model})
            span.end()
            logger.debug(f"[Langfuse] Logged embedding: {name}, texts: {len(texts)}")
            return span
        except Exception as e:
            logger.error(f"[Langfuse] Failed to log embedding: {e}")
            return None

    def log_score(
        self,
        trace,
        name: str,
        value: float,
        comment: str = None,
    ):
        """
        记录评分。

        Parameters
        ----------
        trace : Trace
            Trace 对象

        name : str
            评分名称

        value : float
            评分值

        comment : str, optional
            评论
        """
        if not trace or not self._client:
            return

        try:
            # Langfuse v3: 使用 trace 对象的 score() 方法
            trace.score(
                name=name,
                value=value,
                comment=comment,
            )
            logger.debug(f"[Langfuse] Logged score: {name} = {value}")
        except Exception as e:
            logger.error(f"[Langfuse] Failed to log score: {e}")

    def flush(self):
        """刷新所有待发送的数据。"""
        if self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.error(f"[Langfuse] Flush failed: {e}")

    def shutdown(self):
        """关闭 Langfuse 客户端。"""
        if self._client:
            try:
                self._client.shutdown()
                logger.info("[Langfuse] Shutdown complete")
            except Exception as e:
                logger.error(f"[Langfuse] Shutdown failed: {e}")


# ==================== 装饰器 ====================

def traced(
    name: str = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """
    追踪函数执行的装饰器。

    Parameters
    ----------
    name : str, optional
        追踪名称，默认使用函数名

    capture_input : bool
        是否捕获输入参数

    capture_output : bool
        是否捕获返回值

    Example
    -------
    ```python
    @traced("document_processing")
    def process_document(file_path: str):
        # ...
        return chunks
    ```
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            service = get_langfuse_service()
            if not service.enabled:
                return func(*args, **kwargs)

            trace_name = name or func.__name__

            # 捕获输入
            input_data = None
            if capture_input:
                input_data = {
                    "args": str(args)[:500] if args else None,
                    "kwargs": {
                        k: str(v)[:200] for k, v in kwargs.items()
                    } if kwargs else None,
                }

            trace = service.create_trace(
                name=trace_name,
                input=input_data,
                metadata={"function": func.__name__},
            )

            try:
                result = func(*args, **kwargs)

                # 捕获输出
                output_data = None
                if capture_output and result is not None:
                    output_data = {"result": str(result)[:500]}

                service.end_trace(trace, output=output_data)
                return result

            except Exception as e:
                service.end_trace(
                    trace,
                    output={"error": str(e)},
                    metadata={"status": "error"},
                )
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            service = get_langfuse_service()
            if not service.enabled:
                return await func(*args, **kwargs)

            trace_name = name or func.__name__

            input_data = None
            if capture_input:
                input_data = {
                    "args": str(args)[:500] if args else None,
                    "kwargs": {
                        k: str(v)[:200] for k, v in kwargs.items()
                    } if kwargs else None,
                }

            trace = service.create_trace(
                name=trace_name,
                input=input_data,
                metadata={"function": func.__name__},
            )

            try:
                result = await func(*args, **kwargs)

                output_data = None
                if capture_output and result is not None:
                    output_data = {"result": str(result)[:500]}

                service.end_trace(trace, output=output_data)
                return result

            except Exception as e:
                service.end_trace(
                    trace,
                    output={"error": str(e)},
                    metadata={"status": "error"},
                )
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# ==================== 全局实例 ====================

_langfuse_service: Optional[LangfuseService] = None


def get_langfuse_service() -> LangfuseService:
    """
    获取全局 Langfuse 服务实例。

    Returns
    -------
    LangfuseService
        Langfuse 服务单例
    """
    global _langfuse_service
    if _langfuse_service is None:
        _langfuse_service = LangfuseService()
    return _langfuse_service


def init_langfuse(
    secret_key: str = None,
    public_key: str = None,
    host: str = None,
    enabled: bool = None,
) -> LangfuseService:
    """
    初始化全局 Langfuse 服务。

    Parameters
    ----------
    secret_key : str, optional
        Langfuse Secret Key

    public_key : str, optional
        Langfuse Public Key

    host : str, optional
        Langfuse Host

    enabled : bool, optional
        是否启用

    Returns
    -------
    LangfuseService
        Langfuse 服务实例
    """
    global _langfuse_service
    _langfuse_service = LangfuseService(
        secret_key=secret_key,
        public_key=public_key,
        host=host,
        enabled=enabled,
    )
    return _langfuse_service
