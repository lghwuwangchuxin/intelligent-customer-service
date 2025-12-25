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

from app.agent.state import AgentPhase, ToolCall, ThoughtStep, LongTermMemoryContext
from app.agent.memory import MemoryManager
from app.agent.store import StoreManager
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

# 增强日志工具
from app.utils.log_utils import (
    LogContext,
    AgentLogger,
    log_phase_start,
    log_phase_end,
    log_step,
    log_substep,
    log_tool_call,
    log_timing_summary,
    SEPARATOR_LIGHT,
    SEPARATOR_HEAVY,
)

logger = logging.getLogger(__name__)


# ============== State Definitions ==============

class GraphState(TypedDict):
    """State for the LangGraph agent."""
    # Input
    question: str
    conversation_id: Optional[str]
    user_id: Optional[str]  # For user-specific long-term memory

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

    # Long-term memory
    long_term_memory: Optional[LongTermMemoryContext]
    memory_updates: List[Dict[str, Any]]  # Pending updates to long-term memory

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
    user_id: Optional[str] = None,
    max_iterations: int = 10,
    long_term_memory: Optional[LongTermMemoryContext] = None,
) -> GraphState:
    """Create initial graph state."""
    return GraphState(
        question=question,
        conversation_id=conversation_id,
        user_id=user_id,
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
        long_term_memory=long_term_memory,
        memory_updates=[],
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
        store_manager: Optional[StoreManager] = None,
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
            memory_manager: Memory manager for conversations (short-term).
            store_manager: Store manager for long-term memory (cross-session).
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
        self.store = store_manager  # Long-term memory store
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

        # 记录思考阶段开始
        logger.info(f"[Agent] {SEPARATOR_LIGHT}")
        logger.info(f"[Agent] ▶ 迭代 #{iteration} - 思考阶段 (Think)")

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
            think_start = time.time()
            response = await self.llm.ainvoke(messages)
            think_elapsed = int((time.time() - think_start) * 1000)

            thought, action, action_input = self._parse_thought_response(response)

            # 记录思考结果
            logger.info(f"[Agent]   思考: {thought[:100] if thought else 'None'}...")
            if action and action not in ("回答", "answer"):
                logger.info(f"[Agent]   决策: 执行工具 [{action}] (耗时: {think_elapsed}ms)")
            else:
                logger.info(f"[Agent]   决策: 生成最终回答 (耗时: {think_elapsed}ms)")

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
            logger.error(f"[Agent] Think error: {e}")
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

        # 记录工具调用开始
        logger.info(f"[Agent] ▶ 执行阶段 (Act) - 工具: [{action}]")
        logger.info(f"[Agent]   参数: {json.dumps(action_input, ensure_ascii=False)[:100]}")

        try:
            start_time = time.time()
            result = await self.tools.execute(action, **action_input)
            duration_ms = int((time.time() - start_time) * 1000)

            # 记录工具调用结果
            if result.get("success"):
                result_summary = str(result.get("result", ""))[:100]
                logger.info(f"[Agent] ✓ 工具 [{action}] 执行成功 ({duration_ms}ms)")
                logger.info(f"[Agent]   结果: {result_summary}...")
            else:
                error_msg = result.get("error", "Unknown error")
                logger.warning(f"[Agent] ✗ 工具 [{action}] 执行失败 ({duration_ms}ms): {error_msg}")

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
            logger.error(f"[Agent] ✗ 工具 [{action}] 执行异常: {e}")
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

    # ============== Long-term Memory Methods ==============

    async def _load_long_term_memory(
        self,
        user_id: str,
        question: str,
    ) -> Optional[LongTermMemoryContext]:
        """
        Load long-term memory context for the current session.

        Args:
            user_id: User ID for personalized memory.
            question: Current question for relevance search.

        Returns:
            LongTermMemoryContext with user preferences, entities, and knowledge.
        """
        if not self.store:
            return None

        try:
            # Load user preferences
            user_preferences = {}
            prefs = self.store.get_user_preferences(user_id)
            for pref in prefs:
                if hasattr(pref, 'key') and hasattr(pref, 'value'):
                    user_preferences[pref.key] = pref.value
                elif isinstance(pref, dict):
                    user_preferences[pref.get('key', '')] = pref.get('value')

            # Search for relevant entities (limit to 5)
            relevant_entities = []
            try:
                entities = self.store.search_entities(query=question, limit=5)
                for entity in entities:
                    if hasattr(entity, 'model_dump'):
                        relevant_entities.append(entity.model_dump())
                    elif isinstance(entity, dict):
                        relevant_entities.append(entity)
            except Exception as e:
                logger.debug(f"Entity search failed: {e}")

            # Search for relevant knowledge (limit to 5)
            relevant_knowledge = []
            try:
                knowledge_items = self.store.search_knowledge(query=question, limit=5)
                for item in knowledge_items:
                    if hasattr(item, 'model_dump'):
                        relevant_knowledge.append(item.model_dump())
                    elif isinstance(item, dict):
                        relevant_knowledge.append(item)
            except Exception as e:
                logger.debug(f"Knowledge search failed: {e}")

            return LongTermMemoryContext(
                user_id=user_id,
                user_preferences=user_preferences,
                relevant_entities=relevant_entities,
                relevant_knowledge=relevant_knowledge,
                session_facts={},
            )

        except Exception as e:
            logger.warning(f"Failed to load long-term memory: {e}")
            return None

    async def _persist_memory_updates(
        self,
        user_id: str,
        updates: List[Dict[str, Any]],
    ) -> None:
        """
        Persist queued memory updates to long-term store.

        Args:
            user_id: User ID for user-specific updates.
            updates: List of memory updates to persist.
        """
        if not self.store:
            return

        for update in updates:
            update_type = update.get("type")
            data = update.get("data", {})

            try:
                if update_type == "user_preference":
                    self.store.set_user_preference(
                        user_id=data.get("user_id", user_id),
                        key=data.get("key"),
                        value=data.get("value"),
                        confidence=data.get("confidence", 0.8),
                    )
                elif update_type == "entity":
                    self.store.store_entity(
                        entity_type=data.get("entity_type"),
                        entity_id=data.get("entity_id"),
                        name=data.get("name"),
                        attributes=data.get("attributes", {}),
                    )
                elif update_type == "knowledge":
                    self.store.store_knowledge(
                        topic=data.get("topic"),
                        content=data.get("content"),
                        source=data.get("source"),
                    )
                else:
                    logger.warning(f"Unknown memory update type: {update_type}")
            except Exception as e:
                logger.warning(f"Failed to persist memory update ({update_type}): {e}")

    # ============== Public Methods ==============

    async def run(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the LangGraph agent.

        Args:
            question: User's question.
            conversation_id: Optional conversation ID.
            user_id: Optional user ID for personalized long-term memory.
            history: Optional conversation history.

        Returns:
            Dict with response, tool_calls, thoughts, and plan.
        """
        # Track total execution time
        total_start_time = time.time()

        # 创建增强日志上下文
        request_id = str(uuid.uuid4())[:8]
        log_ctx = LogContext(module="Agent", request_id=request_id, session_id=conversation_id)
        agent_logger = AgentLogger(request_id=request_id)

        # 记录 Agent 运行开始
        agent_logger.start_run(question, self.max_iterations)

        # 记录配置信息
        available_tools = [t.name for t in self.tools.get_all()]
        log_step(log_ctx, "Init", f"LangGraph Agent 启动")
        log_step(log_ctx, "Init", f"可用工具: {available_tools}")
        log_step(log_ctx, "Init", f"最大迭代: {self.max_iterations}, 规划: {self.enable_planning}, 并行工具: {self.enable_parallel_tools}")

        # Load long-term memory context if store is available
        long_term_memory = None
        if self.store and user_id:
            log_step(log_ctx, "LongTermMemory", f"加载用户 [{user_id}] 的长期记忆")
            try:
                long_term_memory = await self._load_long_term_memory(user_id, question)
                if long_term_memory:
                    pref_count = len(long_term_memory.user_preferences)
                    entity_count = len(long_term_memory.relevant_entities)
                    knowledge_count = len(long_term_memory.relevant_knowledge)
                    log_step(log_ctx, "LongTermMemory",
                             f"已加载: 偏好={pref_count}, 实体={entity_count}, 知识={knowledge_count}")
            except Exception as e:
                logger.warning(f"Failed to load long-term memory: {e}")

        # Build messages
        messages = []
        if conversation_id:
            context = self.memory.get_context(conversation_id)
            messages.extend(context)
            log_step(log_ctx, "Memory", f"加载历史上下文: {len(context)} 条消息")
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        # Create initial state with long-term memory
        state = create_graph_state(
            question=question,
            messages=messages,
            conversation_id=conversation_id,
            user_id=user_id,
            max_iterations=self.max_iterations,
            long_term_memory=long_term_memory,
        )

        # Generate session ID for tracing
        session_id = conversation_id or str(uuid.uuid4())

        log_step(log_ctx, "Graph", "开始执行状态图...")

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

                # 记录执行结果
                iterations = final_state.get("iteration", 0)
                tool_calls_count = len(final_state.get("tool_calls", []))
                plan_steps = len(final_state.get("plan", [])) if final_state.get("plan") else 0

                log_step(log_ctx, "Graph", f"状态图执行完成 - 迭代: {iterations}, 工具调用: {tool_calls_count}, 规划步骤: {plan_steps}")

                # 记录思考过程
                for i, thought in enumerate(final_state.get("thoughts", []), 1):
                    thought_text = thought.thought if hasattr(thought, 'thought') else str(thought)
                    action = thought.action if hasattr(thought, 'action') else None
                    agent_logger.log_thinking(i, thought_text, action)

                # 记录工具调用
                for tc in final_state.get("tool_calls", []):
                    tc_name = tc.name if hasattr(tc, 'name') else tc.get('name', 'unknown')
                    tc_result = tc.result if hasattr(tc, 'result') else tc.get('result')
                    tc_error = tc.error if hasattr(tc, 'error') else tc.get('error')
                    tc_duration = tc.duration_ms if hasattr(tc, 'duration_ms') else tc.get('duration_ms', 0)
                    if tc_result:
                        agent_logger.log_observation(tc_name, True, str(tc_result)[:100], tc_duration)
                    elif tc_error:
                        agent_logger.log_observation(tc_name, False, tc_error, tc_duration)

                trace.update(output={
                    "response": final_state.get("final_response"),
                    "iterations": iterations,
                    "tool_calls_count": tool_calls_count,
                    "plan_steps": plan_steps,
                })
            except Exception as e:
                logger.error(f"LangGraph agent error: {e}")
                log_step(log_ctx, "Error", f"Agent 运行错误: {str(e)[:100]}", level="error")
                final_state = {
                    **state,
                    "error": str(e),
                    "final_response": f"处理请求时发生错误: {str(e)}",
                }
                trace.update(output={"error": str(e)})

        # Calculate duration
        total_duration_ms = int((time.time() - total_start_time) * 1000)

        # Prepare result data
        thoughts_data = [
            t.model_dump() if hasattr(t, 'model_dump') else t
            for t in final_state.get("thoughts", [])
        ]
        tool_calls_data = [
            tc.model_dump() if hasattr(tc, 'model_dump') else tc
            for tc in final_state.get("tool_calls", [])
        ]

        # Update memory with messages and interaction record
        if conversation_id:
            log_step(log_ctx, "Memory", "更新对话记忆")
            await self.memory.add_message(conversation_id, "user", question)
            if final_state.get("final_response"):
                await self.memory.add_message(
                    conversation_id, "assistant", final_state["final_response"]
                )

            # Record complete interaction for history viewing
            await self.memory.add_interaction(
                conversation_id=conversation_id,
                interaction_id=request_id,
                question=question,
                response=final_state.get("final_response", ""),
                thoughts=thoughts_data,
                tool_calls=tool_calls_data,
                iterations=final_state.get("iteration", 0),
                duration_ms=total_duration_ms,
                error=final_state.get("error"),
            )

        # Persist long-term memory updates if store is available
        if self.store and user_id:
            memory_updates = final_state.get("memory_updates", [])
            if memory_updates:
                log_step(log_ctx, "LongTermMemory", f"持久化 {len(memory_updates)} 条记忆更新")
                try:
                    await self._persist_memory_updates(user_id, memory_updates)
                except Exception as e:
                    logger.warning(f"Failed to persist memory updates: {e}")

        # 记录运行结束
        agent_logger.end_run(
            success=final_state.get("error") is None,
            iterations=final_state.get("iteration", 0),
            tool_calls=len(final_state.get("tool_calls", []))
        )

        return {
            "response": final_state.get("final_response", "抱歉，我无法回答这个问题。"),
            "thoughts": thoughts_data,
            "tool_calls": tool_calls_data,
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
        user_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Stream LangGraph agent execution.

        Args:
            question: User's question.
            conversation_id: Optional conversation ID.
            user_id: Optional user ID for personalized long-term memory.
            history: Optional conversation history.

        Yields:
            State updates as the graph executes.
        """
        # Load long-term memory context if store is available
        long_term_memory = None
        if self.store and user_id:
            try:
                long_term_memory = await self._load_long_term_memory(user_id, question)
                if long_term_memory:
                    logger.debug(f"Loaded long-term memory for user {user_id}")
            except Exception as e:
                logger.warning(f"Failed to load long-term memory: {e}")

        # Build messages
        messages = []
        if conversation_id:
            context = self.memory.get_context(conversation_id)
            messages.extend(context)
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        # Create initial state with long-term memory
        state = create_graph_state(
            question=question,
            messages=messages,
            conversation_id=conversation_id,
            user_id=user_id,
            max_iterations=self.max_iterations,
            long_term_memory=long_term_memory,
        )

        session_id = conversation_id or str(uuid.uuid4())

        with self.tracer.trace(
            name="langgraph_agent_stream",
            session_id=session_id,
            metadata={"question": question, "user_id": user_id, "streaming": True},
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

                # Persist long-term memory updates if store is available
                if self.store and user_id:
                    memory_updates = final_state.get("memory_updates", [])
                    if memory_updates:
                        try:
                            await self._persist_memory_updates(user_id, memory_updates)
                            logger.debug(f"Persisted {len(memory_updates)} memory updates for user {user_id}")
                        except Exception as e:
                            logger.warning(f"Failed to persist memory updates: {e}")

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
    store_manager: Optional[StoreManager] = None,
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
        memory_manager: Memory manager for conversations (short-term).
        store_manager: Store manager for long-term memory (cross-session).
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
        store_manager=store_manager,
        max_iterations=max_iterations,
        enable_planning=enable_planning,
        enable_parallel_tools=enable_parallel_tools,
        tool_timeout=tool_timeout,
        max_tool_concurrency=max_tool_concurrency,
    )
