"""Metrics collector for monitoring."""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import time

from services.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Metric:
    """Single metric data point."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    name: str
    count: int
    sum: float
    min: float
    max: float
    avg: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Metrics collector for application monitoring.

    Features:
    - Counter, gauge, and histogram metrics
    - Label support
    - Time series storage
    - Prometheus-compatible export
    """

    def __init__(
        self,
        retention_seconds: int = 3600,
        flush_interval: float = 60.0,
    ):
        """
        Initialize metrics collector.

        Args:
            retention_seconds: How long to retain metrics
            flush_interval: Flush interval in seconds
        """
        self.retention_seconds = retention_seconds
        self.flush_interval = flush_interval

        # Metrics storage
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._histograms: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._time_series: Dict[str, List[Metric]] = defaultdict(list)

        self._lock = asyncio.Lock()
        self._cleanup_task = None

    async def start(self):
        """Start the metrics collector."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Metrics collector started")

    async def stop(self):
        """Stop the metrics collector."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    def _labels_key(self, labels: Dict[str, str]) -> str:
        """Convert labels to a hashable key."""
        sorted_items = sorted(labels.items())
        return ",".join(f"{k}={v}" for k, v in sorted_items)

    async def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Increment a counter.

        Args:
            name: Metric name
            value: Value to add
            labels: Metric labels
        """
        labels = labels or {}
        key = self._labels_key(labels)

        async with self._lock:
            self._counters[name][key] += value
            self._time_series[name].append(Metric(
                name=name,
                value=self._counters[name][key],
                labels=labels,
            ))

    async def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Set a gauge value.

        Args:
            name: Metric name
            value: Gauge value
            labels: Metric labels
        """
        labels = labels or {}
        key = self._labels_key(labels)

        async with self._lock:
            self._gauges[name][key] = value
            self._time_series[name].append(Metric(
                name=name,
                value=value,
                labels=labels,
            ))

    async def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Record a histogram value.

        Args:
            name: Metric name
            value: Observed value
            labels: Metric labels
        """
        labels = labels or {}
        key = self._labels_key(labels)

        async with self._lock:
            self._histograms[name][key].append(value)
            self._time_series[name].append(Metric(
                name=name,
                value=value,
                labels=labels,
            ))

    async def time(self, name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.

        Args:
            name: Metric name
            labels: Metric labels

        Returns:
            Context manager
        """
        return TimingContext(self, name, labels)

    async def get_counter(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """
        Get counter value.

        Args:
            name: Metric name
            labels: Metric labels

        Returns:
            Counter value
        """
        labels = labels or {}
        key = self._labels_key(labels)
        return self._counters.get(name, {}).get(key, 0.0)

    async def get_gauge(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """
        Get gauge value.

        Args:
            name: Metric name
            labels: Metric labels

        Returns:
            Gauge value
        """
        labels = labels or {}
        key = self._labels_key(labels)
        return self._gauges.get(name, {}).get(key, 0.0)

    async def get_histogram_summary(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[MetricSummary]:
        """
        Get histogram summary.

        Args:
            name: Metric name
            labels: Metric labels

        Returns:
            MetricSummary or None
        """
        labels = labels or {}
        key = self._labels_key(labels)

        values = self._histograms.get(name, {}).get(key, [])
        if not values:
            return None

        return MetricSummary(
            name=name,
            count=len(values),
            sum=sum(values),
            min=min(values),
            max=max(values),
            avg=sum(values) / len(values),
            labels=labels,
        )

    async def get_time_series(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> List[Metric]:
        """
        Get time series data.

        Args:
            name: Metric name
            start_time: Start time filter
            end_time: End time filter
            labels: Label filter

        Returns:
            List of metrics
        """
        metrics = self._time_series.get(name, [])

        # Filter by time
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]

        # Filter by labels
        if labels:
            metrics = [
                m for m in metrics
                if all(m.labels.get(k) == v for k, v in labels.items())
            ]

        return metrics

    async def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all current metrics.

        Returns:
            Dictionary of all metrics
        """
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: {
                    key: {
                        "count": len(values),
                        "sum": sum(values),
                        "avg": sum(values) / len(values) if values else 0,
                    }
                    for key, values in buckets.items()
                }
                for name, buckets in self._histograms.items()
            },
        }

    async def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus format string
        """
        lines = []

        # Export counters
        for name, buckets in self._counters.items():
            lines.append(f"# TYPE {name} counter")
            for key, value in buckets.items():
                labels_str = f"{{{key}}}" if key else ""
                lines.append(f"{name}{labels_str} {value}")

        # Export gauges
        for name, buckets in self._gauges.items():
            lines.append(f"# TYPE {name} gauge")
            for key, value in buckets.items():
                labels_str = f"{{{key}}}" if key else ""
                lines.append(f"{name}{labels_str} {value}")

        # Export histograms
        for name, buckets in self._histograms.items():
            lines.append(f"# TYPE {name} summary")
            for key, values in buckets.items():
                if values:
                    labels_str = f"{{{key}}}" if key else ""
                    lines.append(f"{name}_count{labels_str} {len(values)}")
                    lines.append(f"{name}_sum{labels_str} {sum(values)}")

        return "\n".join(lines)

    async def record_metrics(self, metrics: List[Metric]):
        """
        Record multiple metrics.

        Args:
            metrics: List of metrics to record
        """
        for metric in metrics:
            # Infer metric type from name
            if metric.name.endswith("_total"):
                await self.increment(metric.name, metric.value, metric.labels)
            elif metric.name.endswith("_seconds") or metric.name.endswith("_ms"):
                await self.histogram(metric.name, metric.value, metric.labels)
            else:
                await self.gauge(metric.name, metric.value, metric.labels)

    async def _cleanup_loop(self):
        """Cleanup old metrics periodically."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._cleanup_old_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def _cleanup_old_metrics(self):
        """Remove old time series data."""
        cutoff = datetime.utcnow()
        cutoff_timestamp = cutoff.timestamp() - self.retention_seconds

        async with self._lock:
            for name in list(self._time_series.keys()):
                self._time_series[name] = [
                    m for m in self._time_series[name]
                    if m.timestamp.timestamp() > cutoff_timestamp
                ]


class TimingContext:
    """Context manager for timing operations."""

    def __init__(
        self,
        collector: MetricsCollector,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.collector = collector
        self.name = name
        self.labels = labels
        self._start_time = None

    async def __aenter__(self):
        self._start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.time() - self._start_time) * 1000  # ms
        await self.collector.histogram(self.name, elapsed, self.labels)
