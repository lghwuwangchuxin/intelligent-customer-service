"""Langfuse collector for observability and tracing."""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import asynccontextmanager
import uuid

from services.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SpanData:
    """Span data for tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    name: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "OK"
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LangfuseCollector:
    """
    Langfuse collector for LLM observability.

    Features:
    - Trace recording
    - Span management
    - LLM call logging
    - Cost tracking
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com",
        enabled: bool = True,
        flush_interval: float = 5.0,
    ):
        """
        Initialize Langfuse collector.

        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL
            enabled: Whether collection is enabled
            flush_interval: Flush interval in seconds
        """
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host
        self.enabled = enabled and public_key and secret_key
        self.flush_interval = flush_interval

        self._langfuse = None
        self._initialized = False
        self._buffer: List[SpanData] = []
        self._flush_task = None

    async def initialize(self):
        """Initialize Langfuse client."""
        if self._initialized or not self.enabled:
            return

        try:
            from langfuse import Langfuse

            self._langfuse = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
            )
            self._initialized = True

            # Start flush task
            self._flush_task = asyncio.create_task(self._flush_loop())

            logger.info("Langfuse collector initialized")
        except ImportError:
            logger.warning("Langfuse package not installed")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self.enabled = False

    async def shutdown(self):
        """Shutdown collector and flush remaining data."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        await self._flush_buffer()

        if self._langfuse:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._langfuse.flush)
            except Exception as e:
                logger.error(f"Error flushing Langfuse: {e}")

    async def record_span(self, span: SpanData):
        """
        Record a span.

        Args:
            span: Span data to record
        """
        if not self.enabled:
            return

        self._buffer.append(span)

    async def record_spans(self, spans: List[SpanData]):
        """
        Record multiple spans.

        Args:
            spans: List of spans to record
        """
        if not self.enabled:
            return

        self._buffer.extend(spans)

    @asynccontextmanager
    async def trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracing.

        Args:
            name: Trace name
            user_id: User ID
            session_id: Session ID
            metadata: Additional metadata

        Yields:
            Trace context
        """
        trace_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        context = TraceContext(
            collector=self,
            trace_id=trace_id,
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
        )

        try:
            yield context
        finally:
            end_time = datetime.utcnow()

            span = SpanData(
                trace_id=trace_id,
                span_id=trace_id,
                name=name,
                start_time=start_time,
                end_time=end_time,
                attributes={
                    "user_id": user_id,
                    "session_id": session_id,
                },
                metadata=metadata or {},
            )

            await self.record_span(span)

    async def log_llm_call(
        self,
        trace_id: str,
        model: str,
        messages: List[Dict[str, str]],
        response: str,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        latency_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log an LLM call.

        Args:
            trace_id: Parent trace ID
            model: Model name
            messages: Input messages
            response: Model response
            tokens_prompt: Prompt tokens
            tokens_completion: Completion tokens
            latency_ms: Latency in milliseconds
            metadata: Additional metadata
        """
        if not self.enabled:
            return

        span = SpanData(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            name=f"llm.{model}",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            attributes={
                "model": model,
                "tokens_prompt": tokens_prompt,
                "tokens_completion": tokens_completion,
                "tokens_total": tokens_prompt + tokens_completion,
                "latency_ms": latency_ms,
            },
            input_data={"messages": messages},
            output_data={"response": response},
            metadata=metadata or {},
        )

        await self.record_span(span)

    async def log_retrieval(
        self,
        trace_id: str,
        query: str,
        documents: List[Dict[str, Any]],
        latency_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a retrieval operation.

        Args:
            trace_id: Parent trace ID
            query: Search query
            documents: Retrieved documents
            latency_ms: Latency in milliseconds
            metadata: Additional metadata
        """
        if not self.enabled:
            return

        span = SpanData(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            name="retrieval",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            attributes={
                "query": query,
                "num_documents": len(documents),
                "latency_ms": latency_ms,
            },
            input_data={"query": query},
            output_data={"documents": documents[:5]},  # Limit for size
            metadata=metadata or {},
        )

        await self.record_span(span)

    async def _flush_loop(self):
        """Background flush loop."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush error: {e}")

    async def _flush_buffer(self):
        """Flush buffered spans to Langfuse."""
        if not self._buffer or not self._langfuse:
            return

        spans = self._buffer.copy()
        self._buffer.clear()

        try:
            loop = asyncio.get_event_loop()

            for span in spans:
                await loop.run_in_executor(
                    None,
                    lambda s=span: self._send_span(s)
                )

            await loop.run_in_executor(None, self._langfuse.flush)

        except Exception as e:
            logger.error(f"Failed to flush spans: {e}")
            # Re-add failed spans
            self._buffer = spans + self._buffer

    def _send_span(self, span: SpanData):
        """Send a span to Langfuse (blocking)."""
        try:
            trace = self._langfuse.trace(
                id=span.trace_id,
                name=span.name,
                metadata=span.metadata,
            )

            if "llm" in span.name:
                trace.generation(
                    id=span.span_id,
                    name=span.name,
                    model=span.attributes.get("model", "unknown"),
                    input=span.input_data,
                    output=span.output_data,
                    usage={
                        "promptTokens": span.attributes.get("tokens_prompt", 0),
                        "completionTokens": span.attributes.get("tokens_completion", 0),
                    },
                    metadata=span.metadata,
                )
            else:
                trace.span(
                    id=span.span_id,
                    name=span.name,
                    input=span.input_data,
                    output=span.output_data,
                    metadata=span.metadata,
                )

        except Exception as e:
            logger.error(f"Failed to send span: {e}")


class TraceContext:
    """Context for a trace."""

    def __init__(
        self,
        collector: LangfuseCollector,
        trace_id: str,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.collector = collector
        self.trace_id = trace_id
        self.name = name
        self.user_id = user_id
        self.session_id = session_id
        self.metadata = metadata or {}

    @asynccontextmanager
    async def span(self, name: str, **kwargs):
        """Create a child span."""
        span_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        try:
            yield span_id
        finally:
            end_time = datetime.utcnow()

            span = SpanData(
                trace_id=self.trace_id,
                span_id=span_id,
                parent_span_id=self.trace_id,
                name=name,
                start_time=start_time,
                end_time=end_time,
                attributes=kwargs,
                metadata=self.metadata,
            )

            await self.collector.record_span(span)

    async def log_llm(self, model: str, messages: List[Dict], response: str, **kwargs):
        """Log LLM call in this trace."""
        await self.collector.log_llm_call(
            trace_id=self.trace_id,
            model=model,
            messages=messages,
            response=response,
            **kwargs,
        )

    async def log_retrieval(self, query: str, documents: List[Dict], **kwargs):
        """Log retrieval in this trace."""
        await self.collector.log_retrieval(
            trace_id=self.trace_id,
            query=query,
            documents=documents,
            **kwargs,
        )
