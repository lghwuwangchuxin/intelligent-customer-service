"""A2A (Agent-to-Agent) protocol client."""

import asyncio
import uuid
import time
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from services.common.logging import get_logger
from ..routing.agent_registry import AgentInfo

logger = get_logger(__name__)


class TaskState(Enum):
    """A2A task state."""
    PENDING = "pending"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class A2AMessage:
    """A2A protocol message."""
    role: str  # user, assistant
    content: str
    parts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class A2ATask:
    """A2A task representation."""
    task_id: str
    agent_id: str
    messages: List[A2AMessage]
    state: TaskState = TaskState.PENDING
    result: Optional[str] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    latency_ms: int = 0


@dataclass
class A2AResponse:
    """Response from A2A agent."""
    task_id: str
    agent_id: str
    agent_name: str
    result: str
    state: TaskState
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0


class A2AClient:
    """
    A2A (Agent-to-Agent) protocol client.

    Implements the A2A protocol for communicating with specialized agents.
    Supports:
    - Synchronous task execution
    - Streaming responses
    - Task cancellation
    - Error handling
    """

    def __init__(
        self,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize A2A client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries

        self._active_tasks: Dict[str, A2ATask] = {}

    async def send_task(
        self,
        agent: AgentInfo,
        message: str,
        context: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> A2AResponse:
        """
        Send a task to an agent.

        Args:
            agent: Target agent
            message: User message
            context: Conversation context
            session_id: Session ID for context

        Returns:
            A2AResponse with result
        """
        import httpx

        task_id = str(uuid.uuid4())
        start_time = time.time()

        # Build A2A request
        request_body = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}],
                },
            },
            "id": task_id,
        }

        if session_id:
            request_body["params"]["sessionId"] = session_id

        if context:
            request_body["params"]["historyLength"] = len(context)

        # Track task
        task = A2ATask(
            task_id=task_id,
            agent_id=agent.id,
            messages=[A2AMessage(role="user", content=message)],
            state=TaskState.PENDING,
        )
        self._active_tasks[task_id] = task

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{agent.url}/a2a",
                    json=request_body,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                result = response.json()

            # Parse response
            if "error" in result:
                task.state = TaskState.FAILED
                task.result = result["error"].get("message", "Unknown error")
            else:
                task_result = result.get("result", {})
                task.state = TaskState(task_result.get("status", {}).get("state", "completed"))
                task.result = self._extract_result(task_result)
                task.artifacts = task_result.get("artifacts", [])

            task.completed_at = datetime.utcnow()
            task.latency_ms = int((time.time() - start_time) * 1000)

            return A2AResponse(
                task_id=task_id,
                agent_id=agent.id,
                agent_name=agent.name,
                result=task.result or "",
                state=task.state,
                artifacts=task.artifacts,
                latency_ms=task.latency_ms,
            )

        except Exception as e:
            logger.error(f"A2A task failed: {e}")
            task.state = TaskState.FAILED
            task.result = str(e)
            task.latency_ms = int((time.time() - start_time) * 1000)

            return A2AResponse(
                task_id=task_id,
                agent_id=agent.id,
                agent_name=agent.name,
                result=f"Error: {e}",
                state=TaskState.FAILED,
                latency_ms=task.latency_ms,
            )

    async def send_task_streaming(
        self,
        agent: AgentInfo,
        message: str,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[A2AResponse]:
        """
        Send a task with streaming response.

        Args:
            agent: Target agent
            message: User message
            session_id: Session ID

        Yields:
            Partial A2AResponse updates
        """
        import httpx

        task_id = str(uuid.uuid4())
        start_time = time.time()

        request_body = {
            "jsonrpc": "2.0",
            "method": "tasks/sendSubscribe",
            "params": {
                "id": task_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}],
                },
            },
            "id": task_id,
        }

        if session_id:
            request_body["params"]["sessionId"] = session_id

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{agent.url}/a2a",
                    json=request_body,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue

                        try:
                            import json
                            data = json.loads(line[5:].strip())

                            result = data.get("result", {})
                            state = TaskState(result.get("status", {}).get("state", "working"))
                            text = self._extract_result(result)

                            yield A2AResponse(
                                task_id=task_id,
                                agent_id=agent.id,
                                agent_name=agent.name,
                                result=text,
                                state=state,
                                latency_ms=int((time.time() - start_time) * 1000),
                            )

                            if state in (TaskState.COMPLETED, TaskState.FAILED):
                                break

                        except Exception as e:
                            logger.warning(f"Failed to parse stream data: {e}")

        except Exception as e:
            logger.error(f"A2A streaming failed: {e}")
            yield A2AResponse(
                task_id=task_id,
                agent_id=agent.id,
                agent_name=agent.name,
                result=f"Error: {e}",
                state=TaskState.FAILED,
                latency_ms=int((time.time() - start_time) * 1000),
            )

    async def send_to_multiple(
        self,
        agents: List[AgentInfo],
        message: str,
        parallel: bool = True,
    ) -> List[A2AResponse]:
        """
        Send task to multiple agents.

        Args:
            agents: List of target agents
            message: User message
            parallel: Execute in parallel

        Returns:
            List of responses
        """
        if parallel:
            tasks = [
                self.send_task(agent, message)
                for agent in agents
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            results = []
            for agent, response in zip(agents, responses):
                if isinstance(response, Exception):
                    results.append(A2AResponse(
                        task_id="",
                        agent_id=agent.id,
                        agent_name=agent.name,
                        result=f"Error: {response}",
                        state=TaskState.FAILED,
                    ))
                else:
                    results.append(response)

            return results
        else:
            results = []
            for agent in agents:
                response = await self.send_task(agent, message)
                results.append(response)
            return results

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: Task ID to cancel

        Returns:
            True if canceled
        """
        if task_id not in self._active_tasks:
            return False

        task = self._active_tasks[task_id]
        task.state = TaskState.CANCELED
        task.completed_at = datetime.utcnow()

        # TODO: Send cancel request to agent
        # For now, just mark as canceled locally

        return True

    def get_task(self, task_id: str) -> Optional[A2ATask]:
        """
        Get task by ID.

        Args:
            task_id: Task ID

        Returns:
            A2ATask or None
        """
        return self._active_tasks.get(task_id)

    def _extract_result(self, task_result: Dict[str, Any]) -> str:
        """Extract text result from task result."""
        # Check for message in result
        if "message" in task_result:
            message = task_result["message"]
            if isinstance(message, dict):
                parts = message.get("parts", [])
                for part in parts:
                    if part.get("type") == "text":
                        return part.get("text", "")
            elif isinstance(message, str):
                return message

        # Check for direct result
        if "result" in task_result:
            return str(task_result["result"])

        return ""
