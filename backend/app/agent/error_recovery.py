"""
Error Recovery Module - Intelligent error handling and recovery strategies.
Provides retry logic, fallback mechanisms, and graceful degradation.
"""
import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from app.agent.prompts import ERROR_RECOVERY_PROMPT

logger = logging.getLogger(__name__)


class RecoveryAction(str, Enum):
    """Available recovery actions."""
    RETRY = "retry"           # Retry with same or modified parameters
    ALTERNATIVE = "alternative"  # Use an alternative tool
    SKIP = "skip"            # Skip this step and continue
    FALLBACK = "fallback"    # Use fallback response
    ESCALATE = "escalate"    # Escalate to user/human


class ErrorCategory(str, Enum):
    """Categories of errors for targeted recovery."""
    TIMEOUT = "timeout"
    NETWORK = "network"
    VALIDATION = "validation"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


@dataclass
class RecoveryConfig:
    """Configuration for error recovery."""
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    backoff_multiplier: float = 2.0
    max_delay: float = 30.0
    enable_llm_recovery: bool = True


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    action_taken: RecoveryAction
    result: Optional[Any] = None
    error: Optional[str] = None
    attempts: int = 0
    total_time_ms: int = 0


@dataclass
class ErrorContext:
    """Context information about an error."""
    error_type: str
    error_message: str
    category: ErrorCategory
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    attempt_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    stack_trace: Optional[str] = None


class ErrorRecoveryManager:
    """
    Manages error recovery strategies for agent operations.

    Features:
    - Automatic error categorization
    - Configurable retry with exponential backoff
    - LLM-based recovery decision making
    - Fallback strategies
    - Error pattern learning
    """

    def __init__(
        self,
        llm_manager=None,
        config: Optional[RecoveryConfig] = None,
        alternative_tools: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize error recovery manager.

        Args:
            llm_manager: LLM manager for intelligent recovery decisions.
            config: Recovery configuration.
            alternative_tools: Mapping of tools to their alternatives.
        """
        self.llm = llm_manager
        self.config = config or RecoveryConfig()
        self.alternative_tools = alternative_tools or {}
        self._error_history: List[ErrorContext] = []

    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error for targeted recovery."""
        error_str = str(error).lower()

        # Timeout errors
        if any(x in error_str for x in ["timeout", "timed out", "time out"]):
            return ErrorCategory.TIMEOUT

        # Network errors
        if any(x in error_str for x in ["connection", "network", "dns", "socket", "refused"]):
            return ErrorCategory.NETWORK

        # Validation errors
        if any(x in error_str for x in ["validation", "invalid", "required", "missing", "type error"]):
            return ErrorCategory.VALIDATION

        # Authorization errors
        if any(x in error_str for x in ["unauthorized", "forbidden", "permission", "access denied", "401", "403"]):
            return ErrorCategory.AUTHORIZATION

        # Not found errors
        if any(x in error_str for x in ["not found", "404", "does not exist", "no such"]):
            return ErrorCategory.NOT_FOUND

        # Rate limit errors
        if any(x in error_str for x in ["rate limit", "too many requests", "429", "quota"]):
            return ErrorCategory.RATE_LIMIT

        # Internal errors
        if any(x in error_str for x in ["internal", "500", "server error"]):
            return ErrorCategory.INTERNAL

        return ErrorCategory.UNKNOWN

    def get_default_strategy(self, category: ErrorCategory) -> RecoveryAction:
        """Get default recovery strategy for error category."""
        strategies = {
            ErrorCategory.TIMEOUT: RecoveryAction.RETRY,
            ErrorCategory.NETWORK: RecoveryAction.RETRY,
            ErrorCategory.VALIDATION: RecoveryAction.ALTERNATIVE,
            ErrorCategory.AUTHORIZATION: RecoveryAction.ESCALATE,
            ErrorCategory.NOT_FOUND: RecoveryAction.ALTERNATIVE,
            ErrorCategory.RATE_LIMIT: RecoveryAction.RETRY,
            ErrorCategory.INTERNAL: RecoveryAction.RETRY,
            ErrorCategory.UNKNOWN: RecoveryAction.FALLBACK,
        }
        return strategies.get(category, RecoveryAction.FALLBACK)

    async def recover(
        self,
        error: Exception,
        tool_name: str,
        tool_args: Dict[str, Any],
        execute_fn: Callable[..., Awaitable[Any]],
        available_tools: Optional[List[str]] = None,
    ) -> RecoveryResult:
        """
        Attempt to recover from an error.

        Args:
            error: The exception that occurred.
            tool_name: Name of the failed tool.
            tool_args: Arguments passed to the tool.
            execute_fn: Function to execute tools.
            available_tools: List of available tool names.

        Returns:
            RecoveryResult with outcome.
        """
        import time
        start_time = time.time()

        # Create error context
        context = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            category=self.categorize_error(error),
            tool_name=tool_name,
            tool_args=tool_args,
        )
        self._error_history.append(context)

        # Get recovery strategy
        action = self.get_default_strategy(context.category)

        # If LLM recovery is enabled and available, get smarter decision
        if self.config.enable_llm_recovery and self.llm:
            action = await self._get_llm_recovery_decision(context, available_tools)

        # Execute recovery action
        result = await self._execute_recovery(
            action=action,
            context=context,
            execute_fn=execute_fn,
            available_tools=available_tools,
        )

        result.total_time_ms = int((time.time() - start_time) * 1000)
        return result

    async def _get_llm_recovery_decision(
        self,
        context: ErrorContext,
        available_tools: Optional[List[str]] = None,
    ) -> RecoveryAction:
        """Use LLM to decide on recovery action."""
        try:
            prompt = ERROR_RECOVERY_PROMPT.format(
                error=context.error_message,
                tool=context.tool_name,
                input=json.dumps(context.tool_args, ensure_ascii=False),
            )

            # Add available alternatives
            if available_tools:
                alternatives = self.alternative_tools.get(context.tool_name, [])
                valid_alternatives = [t for t in alternatives if t in available_tools]
                if valid_alternatives:
                    prompt += f"\n\n可用的替代工具: {', '.join(valid_alternatives)}"

            messages = [
                {"role": "system", "content": "你是一个错误恢复专家，请分析错误并决定最佳恢复策略。"},
                {"role": "user", "content": prompt},
            ]

            response = await self.llm.ainvoke(messages)

            # Parse LLM response
            return self._parse_recovery_decision(response)

        except Exception as e:
            logger.error(f"LLM recovery decision failed: {e}")
            return self.get_default_strategy(context.category)

    def _parse_recovery_decision(self, response: str) -> RecoveryAction:
        """Parse LLM response to extract recovery action."""
        response_lower = response.lower()

        if "retry" in response_lower or "重试" in response_lower:
            return RecoveryAction.RETRY
        elif "alternative" in response_lower or "替代" in response_lower:
            return RecoveryAction.ALTERNATIVE
        elif "skip" in response_lower or "跳过" in response_lower:
            return RecoveryAction.SKIP
        elif "report" in response_lower or "报告" in response_lower or "escalate" in response_lower:
            return RecoveryAction.ESCALATE

        return RecoveryAction.FALLBACK

    async def _execute_recovery(
        self,
        action: RecoveryAction,
        context: ErrorContext,
        execute_fn: Callable[..., Awaitable[Any]],
        available_tools: Optional[List[str]] = None,
    ) -> RecoveryResult:
        """Execute the chosen recovery action."""

        if action == RecoveryAction.RETRY:
            return await self._retry_with_backoff(
                context=context,
                execute_fn=execute_fn,
            )

        elif action == RecoveryAction.ALTERNATIVE:
            return await self._try_alternative(
                context=context,
                execute_fn=execute_fn,
                available_tools=available_tools,
            )

        elif action == RecoveryAction.SKIP:
            return RecoveryResult(
                success=True,
                action_taken=RecoveryAction.SKIP,
                result={"skipped": True, "reason": context.error_message},
            )

        elif action == RecoveryAction.FALLBACK:
            return RecoveryResult(
                success=True,
                action_taken=RecoveryAction.FALLBACK,
                result=self._generate_fallback_response(context),
            )

        else:  # ESCALATE
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.ESCALATE,
                error=f"需要人工处理: {context.error_message}",
            )

    async def _retry_with_backoff(
        self,
        context: ErrorContext,
        execute_fn: Callable[..., Awaitable[Any]],
    ) -> RecoveryResult:
        """Retry operation with exponential backoff."""
        delay = self.config.retry_delay
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                # Wait before retry (except first attempt)
                if attempt > 0:
                    await asyncio.sleep(delay)
                    if self.config.exponential_backoff:
                        delay = min(
                            delay * self.config.backoff_multiplier,
                            self.config.max_delay
                        )

                # Execute
                result = await execute_fn(context.tool_name, **(context.tool_args or {}))

                if result.get("success"):
                    return RecoveryResult(
                        success=True,
                        action_taken=RecoveryAction.RETRY,
                        result=result.get("result"),
                        attempts=attempt + 1,
                    )
                else:
                    last_error = result.get("error", "Unknown error")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")

        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.RETRY,
            error=f"重试 {self.config.max_retries} 次后仍失败: {last_error}",
            attempts=self.config.max_retries,
        )

    async def _try_alternative(
        self,
        context: ErrorContext,
        execute_fn: Callable[..., Awaitable[Any]],
        available_tools: Optional[List[str]] = None,
    ) -> RecoveryResult:
        """Try alternative tools."""
        alternatives = self.alternative_tools.get(context.tool_name, [])

        if available_tools:
            alternatives = [t for t in alternatives if t in available_tools]

        if not alternatives:
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.ALTERNATIVE,
                error=f"没有可用的替代工具用于 {context.tool_name}",
            )

        for alt_tool in alternatives:
            try:
                result = await execute_fn(alt_tool, **(context.tool_args or {}))

                if result.get("success"):
                    return RecoveryResult(
                        success=True,
                        action_taken=RecoveryAction.ALTERNATIVE,
                        result=result.get("result"),
                    )

            except Exception as e:
                logger.warning(f"Alternative tool {alt_tool} failed: {e}")

        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.ALTERNATIVE,
            error="所有替代工具都失败了",
        )

    def _generate_fallback_response(self, context: ErrorContext) -> Dict[str, Any]:
        """Generate a fallback response when recovery fails."""
        return {
            "fallback": True,
            "message": f"操作 {context.tool_name} 失败，使用了回退响应",
            "original_error": context.error_message,
            "category": context.category.value,
        }

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        if not self._error_history:
            return {"total_errors": 0, "categories": {}}

        category_counts = {}
        for ctx in self._error_history:
            cat = ctx.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_errors": len(self._error_history),
            "categories": category_counts,
            "recent_errors": [
                {
                    "tool": ctx.tool_name,
                    "error": ctx.error_message[:100],
                    "category": ctx.category.value,
                    "timestamp": ctx.timestamp.isoformat(),
                }
                for ctx in self._error_history[-5:]
            ],
        }

    def clear_history(self):
        """Clear error history."""
        self._error_history.clear()


# ============== Retry Decorator ==============

def with_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for adding retry logic to async functions.

    Usage:
        @with_retry(max_retries=3, delay=1.0)
        async def my_function():
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {e}"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception

        return wrapper
    return decorator


# ============== Circuit Breaker ==============

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures.

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        async with breaker:
            result = await risky_operation()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit.
            recovery_timeout: Seconds before trying again.
            success_threshold: Successes needed to close circuit.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self._state == CircuitState.OPEN

    async def __aenter__(self):
        """Enter circuit breaker context."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout passed
                import time
                if (time.time() - (self._last_failure_time or 0)) >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                else:
                    raise CircuitBreakerOpen(
                        f"Circuit is open. Retry after {self.recovery_timeout}s"
                    )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit circuit breaker context."""
        async with self._lock:
            if exc_type is None:
                # Success
                self._on_success()
            else:
                # Failure
                self._on_failure()
        return False

    def _on_success(self):
        """Handle successful execution."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        else:
            self._failure_count = 0

    def _on_failure(self):
        """Handle failed execution."""
        import time
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def reset(self):
        """Manually reset circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit is open."""
    pass
