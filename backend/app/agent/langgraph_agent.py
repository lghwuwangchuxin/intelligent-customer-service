"""
LangGraph Agent - State graph based agent implementation.
Uses LangGraph for flexible state management and conditional branching.
"""
import logging
import json
import re
import time
import uuid
from typing import Dict, List, Any, Optional, Annotated, TypedDict, Sequence, Tuple
from operator import add

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

from app.agent.state import AgentPhase, ToolCall, ThoughtStep
from app.agent.memory import MemoryManager
from app.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    THOUGHT_PROMPT,
    FINAL_RESPONSE_PROMPT,
    TASK_PLANNING_PROMPT,
)
from app.mcp.registry import ToolRegistry
from app.core.langfuse_service import get_tracer, LangfuseTracer
from app.agent.parallel_executor import (
    ParallelToolExecutor,
    create_tool_calls_from_result,
    format_parallel_observation,
)
from app.agent.error_recovery import (
    ErrorRecoveryManager,
    RecoveryConfig,
)

logger = logging.getLogger(__name__)


# ============== State Definitions ==============

class GraphState(TypedDict):
    """State for the LangGraph agent."""
    # Input
    question: str
    conversation_id: Optional[str]

    # Messages
    messages: Annotated[Sequence[Dict[str, str]], add]

    # Reasoning
    thoughts: List[ThoughtStep]
    current_thought: Optional[str]

    # Tool execution
    tool_calls: List[ToolCall]
    pending_action: Optional[str]
    pending_action_input: Optional[Dict[str, Any]]
    pending_parallel_actions: Optional[List[Tuple[str, Dict[str, Any]]]]  # For parallel execution
    last_observation: Optional[str]

    # Task planning
    plan: Optional[List[Dict[str, Any]]]
    current_task_index: int

    # Control
    phase: str
    iteration: int
    max_iterations: int
    should_continue: bool

    # Output
    final_response: Optional[str]
    error: Optional[str]


def create_graph_state(
    question: str,
    messages: List[Dict[str, str]],
    conversation_id: Optional[str] = None,
    max_iterations: int = 10,
) -> GraphState:
    """Create initial graph state."""
    return GraphState(
        question=question,
        conversation_id=conversation_id,
        messages=messages,
        thoughts=[],
        current_thought=None,
        tool_calls=[],
        pending_action=None,
        pending_action_input=None,
        pending_parallel_actions=None,
        last_observation=None,
        plan=None,
        current_task_index=0,
        phase=AgentPhase.THINKING.value,
        iteration=0,
        max_iterations=max_iterations,
        should_continue=True,
        final_response=None,
        error=None,
    )


# ============== LangGraph Agent ==============

class LangGraphAgent:
    """
    LangGraph-based Agent with state graph for flexible workflow.

    Features:
    - State graph based execution
    - Conditional branching
    - Task planning support
    - Parallel tool execution
    - Error recovery
    - Full Langfuse tracing
    """

    def __init__(
        self,
        llm_manager,
        tool_registry: Optional[ToolRegistry] = None,
        memory_manager: Optional[MemoryManager] = None,
        max_iterations: int = 10,
        tracer: Optional[LangfuseTracer] = None,
        enable_planning: bool = True,
        enable_parallel_tools: bool = True,
        tool_timeout: float = 30.0,
        max_tool_concurrency: int = 5,
    ):
        """
        Initialize LangGraph agent.

        Args:
            llm_manager: LLM manager for reasoning.
            tool_registry: Registry of available tools.
            memory_manager: Memory manager for conversations.
            max_iterations: Maximum reasoning iterations.
            tracer: Optional Langfuse tracer.
            enable_planning: Enable task planning for complex questions.
            enable_parallel_tools: Enable parallel tool execution.
            tool_timeout: Timeout for tool execution in seconds.
            max_tool_concurrency: Maximum concurrent tool executions.
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is not installed. Install with: pip install langgraph"
            )

        self.llm = llm_manager
        self.tools = tool_registry or ToolRegistry()
        self.memory = memory_manager or MemoryManager(llm_manager=llm_manager)
        self.max_iterations = max_iterations
        self.tracer = tracer or get_tracer()
        self.enable_planning = enable_planning
        self.enable_parallel_tools = enable_parallel_tools

        # Initialize parallel executor
        self.parallel_executor = ParallelToolExecutor(
            tool_registry=self.tools,
            default_timeout=tool_timeout,
            max_concurrency=max_tool_concurrency,
        )

        # Initialize error recovery manager
        self.error_recovery = ErrorRecoveryManager(
            llm_manager=llm_manager,
            config=RecoveryConfig(
                max_retries=3,
                retry_delay=1.0,
                exponential_backoff=True,
                enable_llm_recovery=True,
            ),
            alternative_tools={
                "knowledge_search": ["web_search"],
                "web_search": ["knowledge_search"],
            }
        )

        # Build the state graph
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        # Create the graph
        graph = StateGraph(GraphState)

        # Add nodes
        graph.add_node("plan", self._plan_node)
        graph.add_node("think", self._think_node)
        graph.add_node("act", self._act_node)
        graph.add_node("parallel_act", self._parallel_act_node)  # Parallel execution
        graph.add_node("observe", self._observe_node)
        graph.add_node("respond", self._respond_node)
        graph.add_node("error_handler", self._error_handler_node)

        # Set entry point
        if self.enable_planning:
            graph.set_entry_point("plan")
        else:
            graph.set_entry_point("think")

        # Add edges
        graph.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {
                "think": "think",
                "respond": "respond",
                "error": "error_handler",
            }
        )

        graph.add_conditional_edges(
            "think",
            self._route_after_think,
            {
                "act": "act",
                "parallel_act": "parallel_act",  # Route to parallel execution
                "respond": "respond",
                "error": "error_handler",
                "end": END,
            }
        )

        graph.add_edge("act", "observe")
        graph.add_edge("parallel_act", "observe")  # Parallel also goes to observe

        graph.add_conditional_edges(
            "observe",
            self._route_after_observe,
            {
                "think": "think",
                "respond": "respond",
                "error": "error_handler",
                "end": END,
            }
        )

        graph.add_edge("respond", END)
        graph.add_edge("error_handler", END)

        return graph

    # ============== Graph Nodes ==============

    async def _plan_node(self, state: GraphState) -> GraphState:
        """Plan tasks for complex questions."""
        question = state["question"]

        # Check if question needs planning (simple heuristic)
        needs_planning = (
            len(question) > 100 or
            any(keyword in question.lower() for keyword in [
                "步骤", "流程", "如何", "怎样", "多个", "首先", "然后",
                "step", "process", "how to", "multiple", "first", "then"
            ])
        )

        if not needs_planning:
            return {
                **state,
                "phase": AgentPhase.THINKING.value,
            }

        # Generate plan
        tool_descriptions = self._get_tool_descriptions()
        plan_prompt = TASK_PLANNING_PROMPT.format(
            question=question,
            tools=tool_descriptions,
        )

        messages = [
            {"role": "system", "content": "你是一个任务规划专家，请将复杂问题分解为可执行的步骤。"},
            {"role": "user", "content": plan_prompt},
        ]

        try:
            response = await self.llm.ainvoke(messages)
            plan = self._parse_plan(response)

            return {
                **state,
                "plan": plan,
                "phase": AgentPhase.THINKING.value,
            }
        except Exception as e:
            logger.error(f"Planning error: {e}")
            return {
                **state,
                "phase": AgentPhase.THINKING.value,
            }

    async def _think_node(self, state: GraphState) -> GraphState:
        """Think and decide on action."""
        question = state["question"]
        iteration = state["iteration"] + 1

        # Build observations summary
        observations = ""
        for thought in state["thoughts"]:
            if thought.observation:
                observations += f"- {thought.observation}\n"

        # Get tool descriptions
        tool_descriptions = self._get_tool_descriptions()

        # Build prompt
        system_prompt = AGENT_SYSTEM_PROMPT.format(tools=tool_descriptions)
        thought_prompt = THOUGHT_PROMPT.format(
            question=question,
            observations=observations or "暂无",
        )

        # Add plan context if available
        if state["plan"]:
            current_task = state["plan"][state["current_task_index"]] if state["current_task_index"] < len(state["plan"]) else None
            if current_task:
                thought_prompt += f"\n\n当前任务: {current_task.get('description', '')}"

        messages = [
            {"role": "system", "content": system_prompt},
            *state["messages"],
            {"role": "user", "content": thought_prompt},
        ]

        try:
            response = await self.llm.ainvoke(messages)
            thought, action, action_input = self._parse_thought_response(response)

            # Create thought step
            thought_step = ThoughtStep(
                step=iteration,
                thought=thought,
                action=action,
                action_input=action_input,
            )

            return {
                **state,
                "thoughts": state["thoughts"] + [thought_step],
                "current_thought": thought,
                "pending_action": action,
                "pending_action_input": action_input,
                "iteration": iteration,
                "phase": AgentPhase.ACTING.value if action and action not in ("回答", "answer") else AgentPhase.RESPONDING.value,
            }
        except Exception as e:
            logger.error(f"Think error: {e}")
            return {
                **state,
                "error": str(e),
                "phase": AgentPhase.ERROR.value,
            }

    async def _act_node(self, state: GraphState) -> GraphState:
        """Execute the chosen action/tool."""
        action = state["pending_action"]
        action_input = state["pending_action_input"] or {}

        if not action:
            return {
                **state,
                "phase": AgentPhase.OBSERVING.value,
                "last_observation": "无操作执行",
            }

        try:
            start_time = time.time()
            result = await self.tools.execute(action, **action_input)
            duration_ms = int((time.time() - start_time) * 1000)

            # Record tool call
            tool_call = ToolCall(
                id=f"tc_{state['iteration']}",
                name=action,
                args=action_input,
                result=result.get("result") if result.get("success") else None,
                error=result.get("error"),
                duration_ms=duration_ms,
            )

            return {
                **state,
                "tool_calls": state["tool_calls"] + [tool_call],
                "last_observation": self._format_observation(result),
                "phase": AgentPhase.OBSERVING.value,
            }
        except Exception as e:
            logger.error(f"Act error: {e}")
            return {
                **state,
                "last_observation": f"工具执行错误: {str(e)}",
                "phase": AgentPhase.OBSERVING.value,
            }

    async def _parallel_act_node(self, state: GraphState) -> GraphState:
        """Execute multiple tools in parallel."""
        parallel_actions = state.get("pending_parallel_actions")

        if not parallel_actions:
            return {
                **state,
                "phase": AgentPhase.OBSERVING.value,
                "last_observation": "无并行操作执行",
            }

        try:
            # Execute all tools in parallel
            result = await self.parallel_executor.execute_parallel(
                tool_calls=parallel_actions,
                trace=None,  # Trace is managed at higher level
            )

            # Convert to tool calls
            new_tool_calls = create_tool_calls_from_result(result)

            # Format observation
            observation = format_parallel_observation(result)

            return {
                **state,
                "tool_calls": state["tool_calls"] + new_tool_calls,
                "last_observation": observation,
                "pending_parallel_actions": None,
                "phase": AgentPhase.OBSERVING.value,
            }
        except Exception as e:
            logger.error(f"Parallel act error: {e}")
            return {
                **state,
                "last_observation": f"并行工具执行错误: {str(e)}",
                "pending_parallel_actions": None,
                "phase": AgentPhase.OBSERVING.value,
            }

    async def _observe_node(self, state: GraphState) -> GraphState:
        """Process observation and update thought."""
        observation = state["last_observation"]

        # Update the last thought with observation
        if state["thoughts"]:
            last_thought = state["thoughts"][-1]
            last_thought.observation = observation

        # Check if we should move to next task in plan
        if state["plan"] and state["current_task_index"] < len(state["plan"]) - 1:
            # Simple heuristic: move to next task after successful tool execution
            return {
                **state,
                "current_task_index": state["current_task_index"] + 1,
                "phase": AgentPhase.THINKING.value,
            }

        return {
            **state,
            "phase": AgentPhase.THINKING.value,
        }

    async def _respond_node(self, state: GraphState) -> GraphState:
        """Generate final response."""
        question = state["question"]

        # Build context from thoughts and observations
        thoughts_text = ""
        for i, thought in enumerate(state["thoughts"], 1):
            thoughts_text += f"{i}. {thought.thought}\n"

        observations_text = ""
        for tc in state["tool_calls"]:
            observations_text += f"- {tc.name}: {str(tc.result or tc.error)[:500]}\n"

        prompt = FINAL_RESPONSE_PROMPT.format(
            question=question,
            thoughts=thoughts_text or "直接回答",
            observations=observations_text or "无工具调用",
        )

        messages = [
            {"role": "system", "content": "你是一个专业的客服助手，请基于以下信息生成回答。"},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.llm.ainvoke(messages)
            return {
                **state,
                "final_response": response,
                "phase": AgentPhase.COMPLETED.value,
                "should_continue": False,
            }
        except Exception as e:
            logger.error(f"Respond error: {e}")
            return {
                **state,
                "final_response": "抱歉，生成回答时出现错误。",
                "error": str(e),
                "phase": AgentPhase.ERROR.value,
                "should_continue": False,
            }

    async def _error_handler_node(self, state: GraphState) -> GraphState:
        """Handle errors and attempt recovery."""
        error = state.get("error", "Unknown error")

        # Simple error recovery: generate a response acknowledging the error
        return {
            **state,
            "final_response": f"抱歉，处理您的请求时遇到了问题: {error}",
            "phase": AgentPhase.ERROR.value,
            "should_continue": False,
        }

    # ============== Routing Functions ==============

    def _route_after_plan(self, state: GraphState) -> str:
        """Route after planning."""
        if state.get("error"):
            return "error"
        return "think"

    def _route_after_think(self, state: GraphState) -> str:
        """Route after thinking."""
        if state.get("error"):
            return "error"

        if state["iteration"] >= state["max_iterations"]:
            return "respond"

        # Check for parallel actions first
        parallel_actions = state.get("pending_parallel_actions")
        if parallel_actions and len(parallel_actions) > 1:
            return "parallel_act"

        action = state.get("pending_action")
        if action and action not in ("回答", "answer", None):
            return "act"

        return "respond"

    def _route_after_observe(self, state: GraphState) -> str:
        """Route after observation."""
        if state.get("error"):
            return "error"

        if state["iteration"] >= state["max_iterations"]:
            return "respond"

        if not state.get("should_continue", True):
            return "end"

        return "think"

    # ============== Helper Methods ==============

    def _get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions."""
        tools = self.tools.get_all()
        if not tools:
            return "暂无可用工具"

        descriptions = []
        for tool in tools:
            params = ", ".join([
                f"{p.name}: {p.type}"
                for p in tool.parameters
            ])
            descriptions.append(
                f"- **{tool.name}**: {tool.description}\n"
                f"  参数: {params or '无'}"
            )

        return "\n".join(descriptions)

    def _parse_thought_response(
        self,
        response: str,
    ) -> tuple[str, Optional[str], Optional[Dict]]:
        """Parse the thought/action response from LLM."""
        thought = ""
        action = None
        action_input = None

        # Try to extract thought
        thought_match = re.search(r'思考[：:]\s*(.+?)(?=行动|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            thought = response.split('\n')[0]

        # Try to extract action
        action_match = re.search(r'行动[：:]\s*(\S+)', response)
        if action_match:
            action = action_match.group(1).strip()

        # Try to extract action input
        input_match = re.search(r'行动输入[：:]\s*(\{.+?\})', response, re.DOTALL)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                pass

        # Check for answer action
        if action and action.lower() in ("回答", "answer", "respond", "无"):
            action = "回答"
            action_input = None

        return thought, action, action_input

    def _parse_plan(self, response: str) -> List[Dict[str, Any]]:
        """Parse planning response into task list."""
        plan = []

        # Try to find numbered steps
        step_pattern = r'(\d+)[.、）]\s*(.+?)(?=\d+[.、）]|$)'
        matches = re.findall(step_pattern, response, re.DOTALL)

        for i, (num, desc) in enumerate(matches):
            plan.append({
                "id": i + 1,
                "description": desc.strip(),
                "status": "pending",
            })

        # Fallback: split by newlines
        if not plan:
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            for i, line in enumerate(lines[:10]):  # Limit to 10 tasks
                plan.append({
                    "id": i + 1,
                    "description": line,
                    "status": "pending",
                })

        return plan

    def _format_observation(self, result: Dict[str, Any]) -> str:
        """Format tool result as observation."""
        if result.get("success"):
            data = result.get("result", {})
            if isinstance(data, list):
                return f"找到 {len(data)} 条结果:\n" + "\n".join([
                    f"- {str(item)[:200]}" for item in data[:5]
                ])
            elif isinstance(data, dict):
                return json.dumps(data, ensure_ascii=False, indent=2)[:1000]
            else:
                return str(data)[:1000]
        else:
            return f"工具执行失败: {result.get('error', 'Unknown error')}"

    # ============== Public Methods ==============

    async def run(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the LangGraph agent.

        Args:
            question: User's question.
            conversation_id: Optional conversation ID.
            history: Optional conversation history.

        Returns:
            Dict with response, tool_calls, thoughts, and plan.
        """
        # Build messages
        messages = []
        if conversation_id:
            context = self.memory.get_context(conversation_id)
            messages.extend(context)
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        # Create initial state
        state = create_graph_state(
            question=question,
            messages=messages,
            conversation_id=conversation_id,
            max_iterations=self.max_iterations,
        )

        # Generate session ID for tracing
        session_id = conversation_id or str(uuid.uuid4())

        # Run with tracing
        with self.tracer.trace(
            name="langgraph_agent_run",
            session_id=session_id,
            metadata={
                "question": question,
                "max_iterations": self.max_iterations,
                "planning_enabled": self.enable_planning,
            },
            tags=["agent", "langgraph"],
        ) as trace:
            try:
                # Run the graph
                final_state = await self.compiled_graph.ainvoke(state)

                trace.update(output={
                    "response": final_state.get("final_response"),
                    "iterations": final_state.get("iteration"),
                    "tool_calls_count": len(final_state.get("tool_calls", [])),
                    "plan_steps": len(final_state.get("plan", [])) if final_state.get("plan") else 0,
                })
            except Exception as e:
                logger.error(f"LangGraph agent error: {e}")
                final_state = {
                    **state,
                    "error": str(e),
                    "final_response": f"处理请求时发生错误: {str(e)}",
                }
                trace.update(output={"error": str(e)})

        # Update memory
        if conversation_id:
            await self.memory.add_message(conversation_id, "user", question)
            if final_state.get("final_response"):
                await self.memory.add_message(
                    conversation_id, "assistant", final_state["final_response"]
                )

        return {
            "response": final_state.get("final_response", "抱歉，我无法回答这个问题。"),
            "thoughts": [
                t.model_dump() if hasattr(t, 'model_dump') else t
                for t in final_state.get("thoughts", [])
            ],
            "tool_calls": [
                tc.model_dump() if hasattr(tc, 'model_dump') else tc
                for tc in final_state.get("tool_calls", [])
            ],
            "plan": final_state.get("plan"),
            "conversation_id": conversation_id,
            "iterations": final_state.get("iteration", 0),
            "error": final_state.get("error"),
            "trace_id": session_id,
        }

    async def stream(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Stream LangGraph agent execution.

        Yields state updates as the graph executes.
        """
        # Build messages
        messages = []
        if conversation_id:
            context = self.memory.get_context(conversation_id)
            messages.extend(context)
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        # Create initial state
        state = create_graph_state(
            question=question,
            messages=messages,
            conversation_id=conversation_id,
            max_iterations=self.max_iterations,
        )

        session_id = conversation_id or str(uuid.uuid4())

        with self.tracer.trace(
            name="langgraph_agent_stream",
            session_id=session_id,
            metadata={"question": question, "streaming": True},
            tags=["agent", "langgraph", "streaming"],
        ) as trace:
            try:
                # Stream graph execution
                async for event in self.compiled_graph.astream(state):
                    # Extract node name and state from event
                    for node_name, node_state in event.items():
                        yield {
                            "type": "node",
                            "node": node_name,
                            "phase": node_state.get("phase"),
                            "iteration": node_state.get("iteration"),
                            "thought": node_state.get("current_thought"),
                            "action": node_state.get("pending_action"),
                            "observation": node_state.get("last_observation"),
                        }

                # Get final state
                final_state = await self.compiled_graph.ainvoke(state)

                yield {
                    "type": "response",
                    "content": final_state.get("final_response", ""),
                }

                yield {
                    "type": "done",
                    "iterations": final_state.get("iteration", 0),
                    "tool_calls": len(final_state.get("tool_calls", [])),
                    "trace_id": session_id,
                }

                trace.update(output={
                    "response": final_state.get("final_response"),
                    "iterations": final_state.get("iteration"),
                })

            except Exception as e:
                logger.error(f"LangGraph stream error: {e}")
                yield {
                    "type": "error",
                    "content": str(e),
                }
                trace.update(output={"error": str(e)})


# ============== Factory Function ==============

def create_langgraph_agent(
    llm_manager,
    tool_registry: Optional[ToolRegistry] = None,
    memory_manager: Optional[MemoryManager] = None,
    max_iterations: int = 10,
    enable_planning: bool = True,
    enable_parallel_tools: bool = True,
    tool_timeout: float = 30.0,
    max_tool_concurrency: int = 5,
) -> Optional[LangGraphAgent]:
    """
    Factory function to create LangGraph agent if available.

    Args:
        llm_manager: LLM manager for reasoning.
        tool_registry: Registry of available tools.
        memory_manager: Memory manager for conversations.
        max_iterations: Maximum reasoning iterations.
        enable_planning: Enable task planning for complex questions.
        enable_parallel_tools: Enable parallel tool execution.
        tool_timeout: Timeout for tool execution in seconds.
        max_tool_concurrency: Maximum concurrent tool executions.

    Returns:
        LangGraphAgent instance or None if LangGraph is not installed.
    """
    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available. Install with: pip install langgraph")
        return None

    return LangGraphAgent(
        llm_manager=llm_manager,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        max_iterations=max_iterations,
        enable_planning=enable_planning,
        enable_parallel_tools=enable_parallel_tools,
        tool_timeout=tool_timeout,
        max_tool_concurrency=max_tool_concurrency,
    )
