"""
Parallel Tool Executor - Execute multiple tools concurrently.
Provides error handling, timeout management, and result aggregation.
"""
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.mcp.registry import ToolRegistry
from app.agent.state import ToolCall
from app.core.langfuse_service import get_tracer

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Status of tool execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ToolTask:
    """Represents a single tool execution task."""
    id: str
    tool_name: str
    args: Dict[str, Any]
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_ms: Optional[int] = None


@dataclass
class ParallelExecutionResult:
    """Result of parallel tool execution."""
    tasks: List[ToolTask]
    total_duration_ms: int
    successful_count: int
    failed_count: int
    timeout_count: int

    @property
    def all_success(self) -> bool:
        """Check if all tasks succeeded."""
        return self.failed_count == 0 and self.timeout_count == 0

    def get_results(self) -> Dict[str, Any]:
        """Get results as a dictionary keyed by task ID."""
        return {
            task.id: {
                "tool": task.tool_name,
                "status": task.status.value,
                "result": task.result,
                "error": task.error,
                "duration_ms": task.duration_ms,
            }
            for task in self.tasks
        }

    def get_successful_results(self) -> List[Any]:
        """Get only successful results."""
        return [
            task.result
            for task in self.tasks
            if task.status == ExecutionStatus.SUCCESS and task.result is not None
        ]


class ParallelToolExecutor:
    """
    Execute multiple tools in parallel with error handling.

    Features:
    - Concurrent tool execution
    - Timeout management per tool
    - Error isolation (one failure doesn't affect others)
    - Result aggregation
    - Langfuse tracing
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        default_timeout: float = 30.0,
        max_concurrency: int = 5,
    ):
        """
        Initialize parallel executor.

        Args:
            tool_registry: Registry of available tools.
            default_timeout: Default timeout per tool in seconds.
            max_concurrency: Maximum concurrent tool executions.
        """
        self.tools = tool_registry
        self.default_timeout = default_timeout
        self.max_concurrency = max_concurrency
        self.tracer = get_tracer()
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def execute_parallel(
        self,
        tool_calls: List[Tuple[str, Dict[str, Any]]],
        timeout: Optional[float] = None,
        trace=None,
    ) -> ParallelExecutionResult:
        """
        Execute multiple tools in parallel.

        Args:
            tool_calls: List of (tool_name, args) tuples.
            timeout: Optional timeout override for all tools.
            trace: Optional Langfuse trace context.

        Returns:
            ParallelExecutionResult with all task results.
        """
        timeout = timeout or self.default_timeout
        start_time = time.time()

        # Create tasks
        tasks = [
            ToolTask(
                id=f"parallel_{i}_{tool_name}",
                tool_name=tool_name,
                args=args,
            )
            for i, (tool_name, args) in enumerate(tool_calls)
        ]

        # Execute all tasks concurrently
        if trace:
            with trace.span(
                name="parallel_tool_execution",
                input={"tool_count": len(tasks), "tools": [t.tool_name for t in tasks]},
                metadata={"max_concurrency": self.max_concurrency},
            ) as span:
                await self._execute_all_tasks(tasks, timeout, trace)
                span.end(output={
                    "successful": sum(1 for t in tasks if t.status == ExecutionStatus.SUCCESS),
                    "failed": sum(1 for t in tasks if t.status == ExecutionStatus.FAILED),
                })
        else:
            await self._execute_all_tasks(tasks, timeout, trace)

        total_duration_ms = int((time.time() - start_time) * 1000)

        return ParallelExecutionResult(
            tasks=tasks,
            total_duration_ms=total_duration_ms,
            successful_count=sum(1 for t in tasks if t.status == ExecutionStatus.SUCCESS),
            failed_count=sum(1 for t in tasks if t.status == ExecutionStatus.FAILED),
            timeout_count=sum(1 for t in tasks if t.status == ExecutionStatus.TIMEOUT),
        )

    async def _execute_all_tasks(
        self,
        tasks: List[ToolTask],
        timeout: float,
        trace=None,
    ):
        """Execute all tasks concurrently with semaphore control."""
        async def execute_with_semaphore(task: ToolTask):
            async with self._semaphore:
                await self._execute_single_task(task, timeout, trace)

        await asyncio.gather(
            *[execute_with_semaphore(task) for task in tasks],
            return_exceptions=True,
        )

    async def _execute_single_task(
        self,
        task: ToolTask,
        timeout: float,
        trace=None,
    ):
        """Execute a single tool task with timeout."""
        task.status = ExecutionStatus.RUNNING
        task.start_time = time.time()

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self.tools.execute(task.tool_name, **task.args),
                timeout=timeout,
            )

            task.end_time = time.time()
            task.duration_ms = int((task.end_time - task.start_time) * 1000)

            if result.get("success"):
                task.status = ExecutionStatus.SUCCESS
                task.result = result.get("result")
            else:
                task.status = ExecutionStatus.FAILED
                task.error = result.get("error", "Unknown error")

        except asyncio.TimeoutError:
            task.end_time = time.time()
            task.duration_ms = int((task.end_time - task.start_time) * 1000)
            task.status = ExecutionStatus.TIMEOUT
            task.error = f"Tool execution timed out after {timeout}s"
            logger.warning(f"Tool {task.tool_name} timed out after {timeout}s")

        except asyncio.CancelledError:
            task.end_time = time.time()
            task.status = ExecutionStatus.CANCELLED
            task.error = "Task was cancelled"

        except Exception as e:
            task.end_time = time.time()
            task.duration_ms = int((task.end_time - task.start_time) * 1000)
            task.status = ExecutionStatus.FAILED
            task.error = str(e)
            logger.error(f"Tool {task.tool_name} failed: {e}")

    async def execute_with_dependencies(
        self,
        tool_graph: Dict[str, Dict[str, Any]],
        trace=None,
    ) -> ParallelExecutionResult:
        """
        Execute tools with dependency management.

        Args:
            tool_graph: Dict with tool configs including dependencies.
                Format: {
                    "task_id": {
                        "tool": "tool_name",
                        "args": {...},
                        "depends_on": ["other_task_id"]
                    }
                }
            trace: Optional Langfuse trace context.

        Returns:
            ParallelExecutionResult with all task results.
        """
        start_time = time.time()

        # Create tasks
        tasks_map: Dict[str, ToolTask] = {}
        for task_id, config in tool_graph.items():
            tasks_map[task_id] = ToolTask(
                id=task_id,
                tool_name=config["tool"],
                args=config.get("args", {}),
            )

        # Build dependency graph
        dependencies: Dict[str, List[str]] = {
            task_id: config.get("depends_on", [])
            for task_id, config in tool_graph.items()
        }

        # Execute in topological order
        completed: set = set()
        all_tasks = list(tasks_map.values())

        while len(completed) < len(tasks_map):
            # Find tasks ready to execute (all dependencies completed)
            ready = [
                task_id
                for task_id, deps in dependencies.items()
                if task_id not in completed and all(d in completed for d in deps)
            ]

            if not ready:
                # Circular dependency or error
                for task_id, task in tasks_map.items():
                    if task_id not in completed:
                        task.status = ExecutionStatus.FAILED
                        task.error = "Circular dependency detected"
                break

            # Execute ready tasks in parallel
            ready_tasks = [tasks_map[task_id] for task_id in ready]
            await self._execute_all_tasks(ready_tasks, self.default_timeout, trace)

            completed.update(ready)

            # Pass results to dependent tasks if needed
            for task_id in ready:
                task = tasks_map[task_id]
                if task.status == ExecutionStatus.SUCCESS:
                    # Update args for dependent tasks with results
                    for dep_task_id, config in tool_graph.items():
                        if task_id in config.get("depends_on", []):
                            # Inject result into dependent task args
                            result_key = f"result_from_{task_id}"
                            tasks_map[dep_task_id].args[result_key] = task.result

        total_duration_ms = int((time.time() - start_time) * 1000)

        return ParallelExecutionResult(
            tasks=all_tasks,
            total_duration_ms=total_duration_ms,
            successful_count=sum(1 for t in all_tasks if t.status == ExecutionStatus.SUCCESS),
            failed_count=sum(1 for t in all_tasks if t.status == ExecutionStatus.FAILED),
            timeout_count=sum(1 for t in all_tasks if t.status == ExecutionStatus.TIMEOUT),
        )


# ============== Helper Functions ==============

def create_tool_calls_from_result(
    result: ParallelExecutionResult,
) -> List[ToolCall]:
    """Convert ParallelExecutionResult to list of ToolCall objects."""
    return [
        ToolCall(
            id=task.id,
            name=task.tool_name,
            args=task.args,
            result=task.result,
            error=task.error,
            duration_ms=task.duration_ms,
        )
        for task in result.tasks
    ]


def format_parallel_observation(result: ParallelExecutionResult) -> str:
    """Format parallel execution result as observation text."""
    if result.all_success:
        observations = []
        for task in result.tasks:
            if task.result:
                if isinstance(task.result, list):
                    observations.append(
                        f"[{task.tool_name}] 找到 {len(task.result)} 条结果"
                    )
                elif isinstance(task.result, dict):
                    observations.append(
                        f"[{task.tool_name}] {str(task.result)[:200]}"
                    )
                else:
                    observations.append(
                        f"[{task.tool_name}] {str(task.result)[:200]}"
                    )
        return "\n".join(observations)
    else:
        lines = []
        for task in result.tasks:
            if task.status == ExecutionStatus.SUCCESS:
                lines.append(f"[{task.tool_name}] 成功")
            elif task.status == ExecutionStatus.FAILED:
                lines.append(f"[{task.tool_name}] 失败: {task.error}")
            elif task.status == ExecutionStatus.TIMEOUT:
                lines.append(f"[{task.tool_name}] 超时")
        return "\n".join(lines)
