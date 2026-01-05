"""
Logging configuration module.

Supports:
- Console logging with colored output
- File-based logging with rotation
- Daily rotation (logs/app-YYYY-MM-DD.log)
- Size-based rotation (max 10MB per file, keep 30 backups)
- Separate log files for errors
- JSON format option for structured logging
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import settings


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        result = super().format(record)

        # Restore original levelname
        record.levelname = levelname
        return result


class TimedRotatingFileHandlerWithSize(logging.handlers.TimedRotatingFileHandler):
    """
    Custom handler that combines time-based and size-based rotation.

    - Rotates daily at midnight
    - Also rotates when file exceeds max size
    """

    def __init__(
        self,
        filename: str,
        when: str = 'midnight',
        interval: int = 1,
        backupCount: int = 30,
        encoding: str = 'utf-8',
        maxBytes: int = 10 * 1024 * 1024,  # 10MB default
    ):
        super().__init__(
            filename,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
        )
        self.maxBytes = maxBytes

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        """Check if we should rollover (time-based or size-based)."""
        # First check time-based rotation
        if super().shouldRollover(record):
            return True

        # Then check size-based rotation
        if self.stream is None:
            self.stream = self._open()

        if self.maxBytes > 0:
            msg = self.format(record)
            self.stream.seek(0, 2)  # Seek to end
            if self.stream.tell() + len(msg) >= self.maxBytes:
                return True

        return False


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: Optional[str] = None,
    log_to_console: bool = True,
    log_to_file: bool = True,
    json_format: bool = False,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory for log files (default: logs/)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        json_format: Whether to use JSON format for file logs

    Returns:
        Root logger instance
    """
    # Get settings
    log_dir = log_dir or getattr(settings, 'LOG_DIR', 'logs')
    log_level = log_level or getattr(settings, 'LOG_LEVEL', 'DEBUG' if settings.DEBUG else 'INFO')
    log_to_console = getattr(settings, 'LOG_TO_CONSOLE', log_to_console)
    log_to_file = getattr(settings, 'LOG_TO_FILE', log_to_file)
    json_format = getattr(settings, 'LOG_JSON_FORMAT', json_format)
    max_bytes = getattr(settings, 'LOG_MAX_BYTES', 10 * 1024 * 1024)  # 10MB
    backup_count = getattr(settings, 'LOG_BACKUP_COUNT', 30)

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Get numeric log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console format
    console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    console_formatter = ColoredFormatter(console_format, datefmt='%Y-%m-%d %H:%M:%S')

    # File format
    if json_format:
        file_format = '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s", "filename": "%(filename)s", "lineno": %(lineno)d}'
    else:
        file_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handlers
    if log_to_file:
        # Get current date for log file name
        today = datetime.now().strftime('%Y-%m-%d')

        # Main log file (all levels)
        main_log_file = log_path / f'app-{today}.log'
        main_handler = TimedRotatingFileHandlerWithSize(
            filename=str(main_log_file),
            when='midnight',
            interval=1,
            backupCount=backup_count,
            encoding='utf-8',
            maxBytes=max_bytes,
        )
        main_handler.setLevel(numeric_level)
        main_handler.setFormatter(file_formatter)
        main_handler.suffix = '%Y-%m-%d'
        root_logger.addHandler(main_handler)

        # Error log file (ERROR and CRITICAL only)
        error_log_file = log_path / f'error-{today}.log'
        error_handler = TimedRotatingFileHandlerWithSize(
            filename=str(error_log_file),
            when='midnight',
            interval=1,
            backupCount=backup_count,
            encoding='utf-8',
            maxBytes=max_bytes,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        error_handler.suffix = '%Y-%m-%d'
        root_logger.addHandler(error_handler)

        # RAG-specific log file (for debugging RAG pipeline)
        if getattr(settings, 'LOG_RAG_SEPARATE', True):
            rag_log_file = log_path / f'rag-{today}.log'
            rag_handler = TimedRotatingFileHandlerWithSize(
                filename=str(rag_log_file),
                when='midnight',
                interval=1,
                backupCount=backup_count,
                encoding='utf-8',
                maxBytes=max_bytes,
            )
            rag_handler.setLevel(logging.DEBUG)
            rag_handler.setFormatter(file_formatter)
            rag_handler.suffix = '%Y-%m-%d'

            # Only add to RAG-related loggers
            rag_logger = logging.getLogger('app.rag')
            rag_logger.setLevel(logging.DEBUG)  # Ensure logger level is set
            rag_logger.addHandler(rag_handler)

        # Agent-specific log file (for debugging Agent/ReAct reasoning)
        if getattr(settings, 'LOG_AGENT_SEPARATE', True):
            agent_log_file = log_path / f'agent-{today}.log'
            agent_handler = TimedRotatingFileHandlerWithSize(
                filename=str(agent_log_file),
                when='midnight',
                interval=1,
                backupCount=backup_count,
                encoding='utf-8',
                maxBytes=max_bytes,
            )
            agent_handler.setLevel(logging.DEBUG)
            agent_handler.setFormatter(file_formatter)
            agent_handler.suffix = '%Y-%m-%d'

            # Add to Agent-related loggers
            agent_logger = logging.getLogger('app.agent')
            agent_logger.setLevel(logging.DEBUG)  # Ensure logger level is set
            agent_logger.addHandler(agent_handler)

        # RAG Evaluation-specific log file (for RAGAS evaluation metrics)
        if getattr(settings, 'LOG_RAGAS_SEPARATE', True):
            ragas_log_file = log_path / f'ragas-eval-{today}.log'
            ragas_handler = TimedRotatingFileHandlerWithSize(
                filename=str(ragas_log_file),
                when='midnight',
                interval=1,
                backupCount=backup_count,
                encoding='utf-8',
                maxBytes=max_bytes,
            )
            ragas_handler.setLevel(logging.DEBUG)
            ragas_handler.setFormatter(file_formatter)
            ragas_handler.suffix = '%Y-%m-%d'

            # Add to RAGAS evaluation logger
            # Note: This logger is a child of app.rag, so logs will also appear in rag-*.log
            ragas_eval_logger = logging.getLogger('app.rag.ragas_evaluator')
            ragas_eval_logger.setLevel(logging.DEBUG)  # Ensure logger level is set
            ragas_eval_logger.addHandler(ragas_handler)

    # Reduce noise from third-party libraries
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('aiosqlite').setLevel(logging.WARNING)
    logging.getLogger('langfuse').setLevel(logging.WARNING)
    logging.getLogger('ragas').setLevel(logging.WARNING)

    # Log startup info
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, Dir: {log_path.absolute()}")
    logger.info(f"Console: {log_to_console}, File: {log_to_file}, JSON: {json_format}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def cleanup_old_logs(log_dir: Optional[str] = None, days_to_keep: int = 30) -> int:
    """
    Clean up log files older than specified days.

    Args:
        log_dir: Directory containing log files
        days_to_keep: Number of days to keep logs

    Returns:
        Number of files deleted
    """
    import time

    log_dir = log_dir or getattr(settings, 'LOG_DIR', 'logs')
    log_path = Path(log_dir)

    if not log_path.exists():
        return 0

    deleted_count = 0
    cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)

    for log_file in log_path.glob('*.log*'):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                deleted_count += 1
            except Exception:
                pass

    if deleted_count > 0:
        logger = logging.getLogger(__name__)
        logger.info(f"Cleaned up {deleted_count} old log files")

    return deleted_count
