"""Logging configuration for microservices."""

import logging
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id
        if hasattr(record, "span_id"):
            log_data["span_id"] = record.span_id
        if hasattr(record, "service"):
            log_data["service"] = record.service
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    service_name: str,
    level: str = "INFO",
    log_format: str = "json",
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_dir: str = "/app/logs",
) -> logging.Logger:
    """
    Set up logging for a service.

    Args:
        service_name: Name of the service
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("json" or "text")
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_dir: Directory for log files

    Returns:
        Root logger for the service
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatters
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = ColoredFormatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_path / f"{service_name}.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("grpc").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Create service logger
    service_logger = logging.getLogger(service_name)
    service_logger.info(f"Logging initialized for {service_name}")

    return service_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name."""
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding context to log records."""

    def __init__(
        self,
        logger: logging.Logger,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **extra: Any,
    ):
        self.logger = logger
        self.trace_id = trace_id
        self.span_id = span_id
        self.extra = extra
        self._old_factory = None

    def __enter__(self):
        """Enter context."""
        self._old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            if self.trace_id:
                record.trace_id = self.trace_id
            if self.span_id:
                record.span_id = self.span_id
            if self.extra:
                record.extra_data = self.extra
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    **extra: Any,
):
    """Log a message with additional context."""
    with LogContext(logger, trace_id=trace_id, span_id=span_id, **extra):
        logger.log(level, message)
