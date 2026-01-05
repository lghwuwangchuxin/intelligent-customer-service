"""
Langfuse Observability Service for RAG and LLM monitoring.
Provides tracing, logging, and debugging capabilities.
Updated for Langfuse SDK v3.x API.
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from config.settings import settings

logger = logging.getLogger(__name__)

# Langfuse client - lazy initialization
_langfuse_client = None


def get_langfuse():
    """Get or initialize Langfuse client."""
    global _langfuse_client

    if not settings.LANGFUSE_ENABLED:
        return None

    if _langfuse_client is None:
        if not settings.LANGFUSE_SECRET_KEY or not settings.LANGFUSE_PUBLIC_KEY:
            logger.warning("Langfuse keys not configured. Tracing disabled.")
            return None

        try:
            from langfuse import Langfuse

            _langfuse_client = Langfuse(
                secret_key=settings.LANGFUSE_SECRET_KEY,
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                host=settings.LANGFUSE_HOST,
            )
            logger.info(f"Langfuse initialized: {settings.LANGFUSE_HOST}")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            return None

    return _langfuse_client


class LangfuseTracer:
    """
    Langfuse tracer for monitoring RAG and LLM operations.
    Updated for Langfuse SDK v3.x.

    Usage:
        tracer = LangfuseTracer()
        with tracer.trace("rag_query", user_id="user123") as trace:
            with trace.span("retrieval") as span:
                docs = retrieve_documents(query)
                span.end(output={"doc_count": len(docs)})

            with trace.generation("llm_call", model="qwen3:latest") as gen:
                response = llm.invoke(prompt)
                gen.end(output=response)
    """

    def __init__(self):
        self.langfuse = get_langfuse()
        self._current_trace = None

    @property
    def enabled(self) -> bool:
        """Check if Langfuse is enabled and configured."""
        return self.langfuse is not None

    def trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> "TraceContext":
        """Create a new trace context."""
        return TraceContext(
            tracer=self,
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=tags,
        )

    def flush(self):
        """Flush pending events to Langfuse."""
        if self.enabled:
            try:
                self.langfuse.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse: {e}")


class TraceContext:
    """Context manager for Langfuse traces using v3.x API."""

    def __init__(
        self,
        tracer: LangfuseTracer,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        self.tracer = tracer
        self.name = name
        self.user_id = user_id
        self.session_id = session_id
        self.metadata = metadata or {}
        self.tags = tags or []
        self.trace_id = None
        self.start_time = None
        self._root_span = None

    def __enter__(self) -> "TraceContext":
        self.start_time = datetime.now()
        if self.tracer.enabled and self.tracer.langfuse:
            try:
                # Create a trace ID
                self.trace_id = self.tracer.langfuse.create_trace_id()
                # Build metadata with user_id, session_id, and tags
                full_metadata = {
                    **self.metadata,
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "tags": self.tags,
                }
                # Start the root span using start_span (returns LangfuseSpan directly)
                self._root_span = self.tracer.langfuse.start_span(
                    name=self.name,
                    trace_context={"trace_id": self.trace_id},
                    metadata=full_metadata,
                    input=self.metadata.get("question") or self.metadata.get("message"),
                )
                logger.debug(f"Started trace: {self.name} ({self.trace_id})")
            except Exception as e:
                logger.error(f"Failed to create trace: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._root_span:
            try:
                if exc_type:
                    self._root_span.update(
                        metadata={
                            **self.metadata,
                            "user_id": self.user_id,
                            "session_id": self.session_id,
                            "tags": self.tags,
                            "error": str(exc_val),
                            "error_type": exc_type.__name__ if exc_type else None,
                        },
                        level="ERROR",
                    )
                self._root_span.end()
            except Exception as e:
                logger.error(f"Failed to end trace: {e}")
        self.tracer.flush()
        return False

    def span(
        self,
        name: str,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SpanContext":
        """Create a span within this trace."""
        return SpanContext(
            tracer=self.tracer,
            trace_id=self.trace_id,
            name=name,
            input=input,
            metadata=metadata,
        )

    def generation(
        self,
        name: str,
        model: str,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ) -> "GenerationContext":
        """Create a generation (LLM call) span within this trace."""
        return GenerationContext(
            tracer=self.tracer,
            trace_id=self.trace_id,
            name=name,
            model=model,
            input=input,
            metadata=metadata,
            model_parameters=model_parameters,
        )

    def update(self, output: Optional[Any] = None, **kwargs):
        """Update trace with additional data."""
        if self._root_span:
            try:
                update_data = {}
                if output is not None:
                    update_data["output"] = output
                update_data.update(kwargs)
                self._root_span.update(**update_data)
            except Exception as e:
                logger.error(f"Failed to update trace: {e}")


class SpanContext:
    """Context manager for Langfuse spans using v3.x API."""

    def __init__(
        self,
        tracer: LangfuseTracer,
        trace_id: Optional[str],
        name: str,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.tracer = tracer
        self.trace_id = trace_id
        self.name = name
        self.input = input
        self.metadata = metadata or {}
        self._span = None
        self.start_time = None

    def __enter__(self) -> "SpanContext":
        self.start_time = datetime.now()
        if self.tracer.enabled and self.tracer.langfuse and self.trace_id:
            try:
                # Use trace_context dict for v3.x API
                self._span = self.tracer.langfuse.start_span(
                    name=self.name,
                    trace_context={"trace_id": self.trace_id},
                    input=self.input,
                    metadata=self.metadata,
                )
            except Exception as e:
                logger.error(f"Failed to create span: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span and exc_type:
            try:
                self._span.update(
                    metadata={
                        **self.metadata,
                        "error": str(exc_val),
                    },
                    level="ERROR",
                )
            except Exception as e:
                logger.error(f"Failed to update span with error: {e}")
        return False

    def end(self, output: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        """End the span with output data."""
        if self._span:
            try:
                # In v3.x, update() sets data, end() just marks completion
                if output is not None or metadata:
                    update_data = {}
                    if output is not None:
                        update_data["output"] = output
                    if metadata:
                        update_data["metadata"] = {**self.metadata, **metadata}
                    self._span.update(**update_data)
                self._span.end()
            except Exception as e:
                logger.error(f"Failed to end span: {e}")


class GenerationContext:
    """Context manager for Langfuse generations (LLM calls) using v3.x API."""

    def __init__(
        self,
        tracer: LangfuseTracer,
        trace_id: Optional[str],
        name: str,
        model: str,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ):
        self.tracer = tracer
        self.trace_id = trace_id
        self.name = name
        self.model = model
        self.input = input
        self.metadata = metadata or {}
        self.model_parameters = model_parameters or {}
        self._generation = None
        self.start_time = None

    def __enter__(self) -> "GenerationContext":
        self.start_time = datetime.now()
        if self.tracer.enabled and self.tracer.langfuse and self.trace_id:
            try:
                # Use trace_context dict for v3.x API
                self._generation = self.tracer.langfuse.start_generation(
                    name=self.name,
                    trace_context={"trace_id": self.trace_id},
                    model=self.model,
                    input=self.input,
                    metadata=self.metadata,
                    model_parameters=self.model_parameters,
                )
            except Exception as e:
                logger.error(f"Failed to create generation: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._generation and exc_type:
            try:
                self._generation.update(
                    metadata={
                        **self.metadata,
                        "error": str(exc_val),
                    },
                    level="ERROR",
                )
            except Exception as e:
                logger.error(f"Failed to update generation with error: {e}")
        return False

    def end(
        self,
        output: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """End the generation with output and usage data."""
        if self._generation:
            try:
                # In v3.x, update() sets data, end() just marks completion
                if output is not None or usage or metadata:
                    update_data = {}
                    if output is not None:
                        update_data["output"] = output
                    if usage:
                        update_data["usage_details"] = usage  # v3.x uses usage_details
                    if metadata:
                        update_data["metadata"] = {**self.metadata, **metadata}
                    self._generation.update(**update_data)
                self._generation.end()
            except Exception as e:
                logger.error(f"Failed to end generation: {e}")


# Global tracer instance
_tracer: Optional[LangfuseTracer] = None


def get_tracer() -> LangfuseTracer:
    """Get the global Langfuse tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = LangfuseTracer()
    return _tracer
