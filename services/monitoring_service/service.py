"""Monitoring Service - Main service implementation."""

from typing import List, Dict, Any, Optional
from datetime import datetime

from services.common.logging import get_logger
from services.common.config import get_service_config

from .collectors import LangfuseCollector, MetricsCollector
from .collectors.langfuse_collector import SpanData
from .collectors.metrics_collector import Metric

logger = get_logger(__name__)


class MonitoringService:
    """
    Monitoring Service for observability.

    Features:
    - Trace recording (Langfuse)
    - Metrics collection
    - Span management
    - Export in various formats
    """

    def __init__(
        self,
        langfuse_collector: Optional[LangfuseCollector] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        """
        Initialize monitoring service.

        Args:
            langfuse_collector: Langfuse collector instance
            metrics_collector: Metrics collector instance
        """
        self.langfuse = langfuse_collector or LangfuseCollector()
        self.metrics = metrics_collector or MetricsCollector()

        self._initialized = False

    @classmethod
    def from_config(cls, config=None) -> "MonitoringService":
        """
        Create service from configuration.

        Args:
            config: Service configuration

        Returns:
            MonitoringService instance
        """
        if config is None:
            config = get_service_config("monitoring-service")

        langfuse_collector = LangfuseCollector(
            public_key=getattr(config, 'langfuse_public_key', None),
            secret_key=getattr(config, 'langfuse_secret_key', None),
            host=getattr(config, 'langfuse_host', 'https://cloud.langfuse.com'),
        )

        metrics_collector = MetricsCollector()

        return cls(
            langfuse_collector=langfuse_collector,
            metrics_collector=metrics_collector,
        )

    async def initialize(self):
        """Initialize service."""
        if self._initialized:
            return

        await self.langfuse.initialize()
        await self.metrics.start()

        self._initialized = True
        logger.info("Monitoring Service initialized")

    async def shutdown(self):
        """Shutdown service."""
        await self.langfuse.shutdown()
        await self.metrics.stop()
        self._initialized = False
        logger.info("Monitoring Service shutdown")

    async def record_trace(self, spans: List[Dict[str, Any]]) -> bool:
        """
        Record trace spans.

        Args:
            spans: List of span dictionaries

        Returns:
            True if successful
        """
        try:
            span_data = [
                SpanData(
                    trace_id=s["trace_id"],
                    span_id=s["span_id"],
                    parent_span_id=s.get("parent_span_id"),
                    name=s.get("name", ""),
                    start_time=datetime.fromisoformat(s["start_time"]) if s.get("start_time") else None,
                    end_time=datetime.fromisoformat(s["end_time"]) if s.get("end_time") else None,
                    attributes=s.get("attributes", {}),
                    status=s.get("status", "OK"),
                )
                for s in spans
            ]

            await self.langfuse.record_spans(span_data)
            return True

        except Exception as e:
            logger.error(f"Failed to record trace: {e}")
            return False

    async def record_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        """
        Record metrics.

        Args:
            metrics: List of metric dictionaries

        Returns:
            True if successful
        """
        try:
            metric_data = [
                Metric(
                    name=m["name"],
                    value=m["value"],
                    labels=m.get("labels", {}),
                    timestamp=datetime.fromisoformat(m["timestamp"]) if m.get("timestamp") else datetime.utcnow(),
                )
                for m in metrics
            ]

            await self.metrics.record_metrics(metric_data)
            return True

        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")
            return False

    async def get_metrics(
        self,
        name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Get metrics.

        Args:
            name: Metric name filter
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            labels: Label filter

        Returns:
            Dictionary with metrics
        """
        if name:
            start_dt = datetime.fromisoformat(start_time) if start_time else None
            end_dt = datetime.fromisoformat(end_time) if end_time else None

            time_series = await self.metrics.get_time_series(
                name=name,
                start_time=start_dt,
                end_time=end_dt,
                labels=labels,
            )

            return {
                "name": name,
                "metrics": [
                    {
                        "value": m.value,
                        "labels": m.labels,
                        "timestamp": m.timestamp.isoformat(),
                    }
                    for m in time_series
                ],
            }

        # Return all metrics summary
        all_metrics = await self.metrics.get_all_metrics()
        return all_metrics

    async def get_prometheus_metrics(self) -> str:
        """
        Get metrics in Prometheus format.

        Returns:
            Prometheus format string
        """
        return await self.metrics.export_prometheus()

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
            trace_id: Trace ID
            model: Model name
            messages: Input messages
            response: Model response
            tokens_prompt: Prompt tokens
            tokens_completion: Completion tokens
            latency_ms: Latency in ms
            metadata: Additional metadata
        """
        await self.langfuse.log_llm_call(
            trace_id=trace_id,
            model=model,
            messages=messages,
            response=response,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            latency_ms=latency_ms,
            metadata=metadata,
        )

        # Record metrics
        await self.metrics.increment(
            "llm_calls_total",
            labels={"model": model},
        )
        await self.metrics.histogram(
            "llm_latency_ms",
            latency_ms,
            labels={"model": model},
        )
        await self.metrics.increment(
            "llm_tokens_total",
            tokens_prompt + tokens_completion,
            labels={"model": model, "type": "total"},
        )

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
            trace_id: Trace ID
            query: Search query
            documents: Retrieved documents
            latency_ms: Latency in ms
            metadata: Additional metadata
        """
        await self.langfuse.log_retrieval(
            trace_id=trace_id,
            query=query,
            documents=documents,
            latency_ms=latency_ms,
            metadata=metadata,
        )

        # Record metrics
        await self.metrics.increment("retrieval_calls_total")
        await self.metrics.histogram("retrieval_latency_ms", latency_ms)
        await self.metrics.histogram("retrieval_documents_count", len(documents))

    async def health_check(self) -> Dict[str, Any]:
        """
        Check service health.

        Returns:
            Health status
        """
        return {
            "status": "healthy",
            "langfuse_enabled": self.langfuse.enabled,
            "metrics_collector": "running" if self._initialized else "stopped",
        }
