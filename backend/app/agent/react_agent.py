"""
ReAct Agent - Reasoning and Acting agent with tool calling support.
Implements the ReAct pattern: Think -> Act -> Observe -> Repeat
With full Langfuse tracing integration for observability.

日志追踪结构
-----------
```
Trace: agent_run
├── Phase: Agent Initialization (配置、工具列表)
├── Phase: Memory Loading (历史上下文)
├── Loop: ReAct Iterations
│   ├── Step: Think (思考决策)
│   │   ├── LLM调用统计
│   │   └── 解析结果
│   ├── Step: Act (工具执行)
│   │   ├── 工具参数
│   │   ├── 执行耗时
│   │   └── 结果摘要
│   └── Step: Observe (结果观察)
├── Phase: Response Generation
└── Phase: Memory Update
```
"""
import logging
import json
import re
import time
import uuid
from typing import Dict, List, Any, Optional, AsyncIterator

from app.agent.state import (
    AgentState,
    ToolCall,
    create_initial_state,
    AgentStateManager,
)
from app.agent.memory import MemoryManager
from app.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    THOUGHT_PROMPT,
    FINAL_RESPONSE_PROMPT,
)
from app.mcp.registry import ToolRegistry
from app.core.langfuse_service import get_tracer, LangfuseTracer

# 增强日志工具
from app.utils.log_utils import (
    LogContext,
    AgentLogger,
    log_phase_start,
    log_phase_end,
    log_step,
    log_substep,
    log_llm_call,
    log_timing_summary,
    log_memory_operation,
    SEPARATOR_LIGHT,
    SEPARATOR_HEAVY,
)

logger = logging.getLogger(__name__)


class ReActAgent:
    """
    ReAct Agent with tool calling and memory support.

    Features:
    - ReAct reasoning loop (Think -> Act -> Observe)
    - Multi-tool support via MCP registry
    - Conversation memory with summarization
    - Streaming response support
    - Error handling and recovery
    - Full Langfuse tracing for observability
    """

    def __init__(
        self,
        llm_manager,
        tool_registry: Optional[ToolRegistry] = None,
        memory_manager: Optional[MemoryManager] = None,
        max_iterations: int = 10,
        tracer: Optional[LangfuseTracer] = None,
    ):
        """
        Initialize ReAct agent.

        Args:
            llm_manager: LLM manager for reasoning.
            tool_registry: Registry of available tools.
            memory_manager: Memory manager for conversations.
            max_iterations: Maximum reasoning iterations.
            tracer: Optional Langfuse tracer for observability.
        """
        self.llm = llm_manager
        self.tools = tool_registry or ToolRegistry()
        self.memory = memory_manager or MemoryManager(llm_manager=llm_manager)
        self.max_iterations = max_iterations
        self.tracer = tracer or get_tracer()

    async def run(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the agent on a question with full Langfuse tracing.

        Args:
            question: User's question.
            conversation_id: Optional conversation ID for memory.
            history: Optional conversation history.
            trace_id: Optional trace ID for linking to existing trace.

        Returns:
            Dict with response, tool_calls, and thoughts.
        """
        # 创建增强日志上下文
        request_id = str(uuid.uuid4())[:8]
        log_ctx = LogContext(module="Agent", request_id=request_id, session_id=conversation_id)
        agent_logger = AgentLogger(request_id=request_id)

        # 记录Agent运行开始
        agent_logger.start_run(question, self.max_iterations)

        # 记录配置信息
        available_tools = [t.name for t in self.tools.get_all()]
        log_step(log_ctx, "Init", f"可用工具: {available_tools}")
        log_step(log_ctx, "Init", f"最大迭代次数: {self.max_iterations}")

        # Build initial messages
        messages = self._build_initial_messages(question, history, conversation_id)
        log_step(log_ctx, "Memory", f"加载历史上下文: {len(messages)} 条消息")
        if conversation_id:
            memory = self.memory.get_memory(conversation_id)
            has_summary = memory.summary is not None if memory else False
            log_memory_operation(log_ctx, "LoadContext", conversation_id, len(messages), has_summary)

        # Create initial state
        state = create_initial_state(
            messages=messages,
            conversation_id=conversation_id,
            max_iterations=self.max_iterations,
        )

        # Generate session ID for tracing
        session_id = conversation_id or str(uuid.uuid4())

        # Run with Langfuse tracing
        log_step(log_ctx, "ReAct", "开始 ReAct 推理循环")
        with self.tracer.trace(
            name="agent_run",
            session_id=session_id,
            metadata={
                "question": question,
                "max_iterations": self.max_iterations,
                "tools_available": available_tools,
            },
            tags=["agent", "react"],
        ) as trace:
            try:
                state = await self._run_loop(state, question, trace, log_ctx, agent_logger)

                # 记录循环完成
                log_step(log_ctx, "ReAct", f"推理循环完成 - 迭代: {state['current_step']}, 工具调用: {len(state['tool_calls'])}")

                trace.update(
                    output={
                        "response": state["final_response"],
                        "iterations": state["current_step"],
                        "tool_calls_count": len(state["tool_calls"]),
                    }
                )
            except Exception as e:
                logger.error(f"[Agent] [{request_id}] 运行错误: {e}", exc_info=True)
                log_step(log_ctx, "Error", f"Agent运行异常: {str(e)[:100]}", level="error")
                state = AgentStateManager.set_error(state, str(e))
                trace.update(output={"error": str(e)})

        # Update memory
        if conversation_id:
            log_step(log_ctx, "Memory", "更新对话记忆")
            await self.memory.add_message(conversation_id, "user", question)
            if state["final_response"]:
                await self.memory.add_message(
                    conversation_id, "assistant", state["final_response"]
                )
            log_memory_operation(log_ctx, "SaveContext", conversation_id,
                                len(self.memory.get_memory(conversation_id).messages) if self.memory.get_memory(conversation_id) else 0,
                                self.memory.get_memory(conversation_id).summary is not None if self.memory.get_memory(conversation_id) else False)

        # 记录运行结束和耗时统计
        agent_logger.end_run(
            success=state.get("error") is None,
            iterations=state["current_step"],
            tool_calls=len(state["tool_calls"])
        )

        return {
            "response": state["final_response"] or "抱歉，我无法回答这个问题。",
            "thoughts": [t.model_dump() if hasattr(t, 'model_dump') else t for t in state["thoughts"]],
            "tool_calls": [tc.model_dump() if hasattr(tc, 'model_dump') else tc for tc in state["tool_calls"]],
            "conversation_id": conversation_id,
            "iterations": state["current_step"],
            "error": state.get("error"),
            "trace_id": session_id,
        }

    async def stream(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream agent response with intermediate steps and Langfuse tracing.

        Yields:
            Dicts with type (thought/action/observation/response) and content.
        """
        messages = self._build_initial_messages(question, history, conversation_id)
        state = create_initial_state(
            messages=messages,
            conversation_id=conversation_id,
            max_iterations=self.max_iterations,
        )

        # Generate session ID for tracing
        session_id = conversation_id or str(uuid.uuid4())

        # Create trace context for streaming
        trace_context = self.tracer.trace(
            name="agent_stream",
            session_id=session_id,
            metadata={
                "question": question,
                "max_iterations": self.max_iterations,
                "streaming": True,
            },
            tags=["agent", "react", "streaming"],
        )

        with trace_context as trace:
            iteration = 0
            while AgentStateManager.should_continue(state) and iteration < self.max_iterations:
                iteration += 1

                # Think
                yield {"type": "status", "content": "思考中...", "step": iteration}

                with trace.span(
                    name=f"think_step_{iteration}",
                    input={"question": question, "iteration": iteration},
                ) as think_span:
                    thought, action, action_input = await self._think(state, question, trace)
                    think_span.end(output={"thought": thought, "action": action})

                yield {
                    "type": "thought",
                    "content": thought,
                    "step": iteration,
                }

                state = AgentStateManager.add_thought(state, thought)

                # Check if we should respond
                if action == "回答" or action == "answer" or action is None:
                    with trace.generation(
                        name="generate_response",
                        model=getattr(self.llm, 'model', 'unknown'),
                    ) as gen:
                        response = await self._generate_response(state, question)
                        gen.end(output=response)

                    state = AgentStateManager.set_final_response(state, response)

                    # Stream the response
                    yield {"type": "response_start", "step": iteration}
                    yield {"type": "response", "content": response}
                    yield {"type": "response_end", "step": iteration}
                    break

                # Act
                state = AgentStateManager.record_action(state, action, action_input or {})
                yield {
                    "type": "action",
                    "tool": action,
                    "input": action_input,
                    "step": iteration,
                }

                # Execute tool with tracing
                start_time = time.time()
                with trace.span(
                    name=f"tool_call_{action}",
                    input={"tool": action, "args": action_input},
                ) as tool_span:
                    result = await self.tools.execute(action, **(action_input or {}))
                    duration_ms = int((time.time() - start_time) * 1000)
                    tool_span.end(output={
                        "success": result.get("success"),
                        "duration_ms": duration_ms,
                    })

                # Record tool call
                tool_call = ToolCall(
                    id=f"tc_{iteration}",
                    name=action,
                    args=action_input or {},
                    result=result.get("result") if result.get("success") else None,
                    error=result.get("error"),
                    duration_ms=duration_ms,
                )
                state = AgentStateManager.add_tool_call(state, tool_call)

                # Observe
                observation = self._format_observation(result)
                state = AgentStateManager.record_observation(state, observation)

                yield {
                    "type": "observation",
                    "content": observation,
                    "success": result.get("success", False),
                    "step": iteration,
                }

            # Update trace with final output
            trace.update(output={
                "response": state.get("final_response"),
                "iterations": iteration,
                "tool_calls_count": len(state["tool_calls"]),
            })

        # Update memory
        if conversation_id and state["final_response"]:
            await self.memory.add_message(conversation_id, "user", question)
            await self.memory.add_message(
                conversation_id, "assistant", state["final_response"]
            )

        yield {
            "type": "done",
            "iterations": iteration,
            "tool_calls": len(state["tool_calls"]),
            "trace_id": session_id,
        }

    async def _run_loop(
        self,
        state: AgentState,
        question: str,
        trace: Optional[Any] = None,
        log_ctx: Optional[LogContext] = None,
        agent_logger: Optional[AgentLogger] = None,
    ) -> AgentState:
        """Run the main ReAct loop with Langfuse tracing and enhanced logging."""
        iteration = 0

        # 创建默认日志上下文
        if not log_ctx:
            log_ctx = LogContext(module="Agent")
        if not agent_logger:
            agent_logger = AgentLogger()

        prefix = log_ctx.get_prefix()
        logger.info(f"{prefix} {SEPARATOR_LIGHT}")
        logger.info(f"{prefix} 进入 ReAct 主循环 (最大迭代: {self.max_iterations})")

        while AgentStateManager.should_continue(state) and iteration < self.max_iterations:
            iteration += 1
            iteration_start = time.time()

            # 记录迭代开始
            logger.info(f"{prefix} {SEPARATOR_LIGHT}")
            logger.info(f"{prefix} ▶ 迭代 #{iteration}/{self.max_iterations}")

            # Think: Decide what to do - with tracing
            think_start = time.time()
            log_step(log_ctx, f"Iter{iteration}", "步骤1: 思考 (Think) - 分析问题并决策")

            if trace:
                with trace.span(
                    name=f"think_step_{iteration}",
                    input={"question": question, "iteration": iteration},
                    metadata={"phase": "thinking"},
                ) as think_span:
                    thought, action, action_input = await self._think(state, question, trace)
                    think_span.end(output={
                        "thought": thought,
                        "action": action,
                        "action_input": action_input,
                    })
            else:
                thought, action, action_input = await self._think(state, question, trace)

            think_elapsed = int((time.time() - think_start) * 1000)

            # 记录思考结果
            agent_logger.log_thinking(iteration, thought or "", action)
            log_substep(log_ctx, f"Iter{iteration}", "Think",
                       f"思考: {thought[:80] if thought else 'None'}...")
            log_substep(log_ctx, f"Iter{iteration}", "Decision",
                       f"决策: action={action}, 耗时={think_elapsed}ms")
            log_ctx.record_step_time(f"Think_{iteration}", think_elapsed)

            state = AgentStateManager.add_thought(state, thought)

            # Check if we should respond directly
            if action == "回答" or action == "answer" or action is None:
                log_step(log_ctx, f"Iter{iteration}", "决定直接回答 - 生成最终响应")

                response_start = time.time()
                if trace:
                    with trace.generation(
                        name="generate_response",
                        model=getattr(self.llm, 'model', 'unknown'),
                        input={"question": question, "thoughts": len(state["thoughts"])},
                        metadata={"phase": "responding"},
                    ) as gen:
                        response = await self._generate_response(state, question)
                        gen.end(output=response)
                else:
                    response = await self._generate_response(state, question)

                response_elapsed = int((time.time() - response_start) * 1000)
                agent_logger.log_response(len(response), response_elapsed)
                log_ctx.record_step_time("ResponseGeneration", response_elapsed)

                state = AgentStateManager.set_final_response(state, response)
                logger.info(f"{prefix}   响应生成完成 ({len(response)} 字符, {response_elapsed}ms)")
                break

            # Act: Execute the chosen tool - with tracing
            log_step(log_ctx, f"Iter{iteration}", f"步骤2: 执行 (Act) - 调用工具 [{action}]")
            agent_logger.log_action(action, action_input or {})

            state = AgentStateManager.record_action(state, action, action_input or {})

            start_time = time.time()
            if trace:
                with trace.span(
                    name=f"tool_call_{action}",
                    input={"tool": action, "args": action_input},
                    metadata={"phase": "acting", "iteration": iteration},
                ) as tool_span:
                    result = await self.tools.execute(action, **(action_input or {}))
                    duration_ms = int((time.time() - start_time) * 1000)
                    tool_span.end(output={
                        "success": result.get("success"),
                        "result": str(result.get("result", ""))[:500],
                        "error": result.get("error"),
                        "duration_ms": duration_ms,
                    })
            else:
                result = await self.tools.execute(action, **(action_input or {}))
                duration_ms = int((time.time() - start_time) * 1000)

            # Record tool call
            tool_call = ToolCall(
                id=f"tc_{iteration}",
                name=action,
                args=action_input or {},
                result=result.get("result") if result.get("success") else None,
                error=result.get("error"),
                duration_ms=duration_ms,
            )
            state = AgentStateManager.add_tool_call(state, tool_call)

            # 记录工具调用结果
            if result.get("success"):
                result_summary = str(result.get("result", ""))[:100]
                agent_logger.log_observation(action, True, result_summary, duration_ms)
            else:
                error_msg = result.get("error", "Unknown error")
                agent_logger.log_observation(action, False, error_msg, duration_ms)

            log_ctx.record_step_time(f"Tool_{action}_{iteration}", duration_ms)

            # Observe: Process the result
            log_step(log_ctx, f"Iter{iteration}", "步骤3: 观察 (Observe) - 处理工具结果")
            observation = self._format_observation(result)
            state = AgentStateManager.record_observation(state, observation)

            log_substep(log_ctx, f"Iter{iteration}", "Observation",
                       f"观察结果: {observation[:100]}...")

            iteration_elapsed = int((time.time() - iteration_start) * 1000)
            logger.info(f"{prefix} ◀ 迭代 #{iteration} 完成 ({iteration_elapsed}ms)")

        # If we've exhausted iterations, generate a response anyway
        if not state["final_response"]:
            log_step(log_ctx, "Fallback", f"达到最大迭代次数 ({self.max_iterations})，生成回退响应")

            response_start = time.time()
            if trace:
                with trace.generation(
                    name="generate_fallback_response",
                    model=getattr(self.llm, 'model', 'unknown'),
                    input={"question": question, "reason": "max_iterations_reached"},
                    metadata={"phase": "responding", "fallback": True},
                ) as gen:
                    response = await self._generate_response(state, question)
                    gen.end(output=response)
            else:
                response = await self._generate_response(state, question)

            response_elapsed = int((time.time() - response_start) * 1000)
            agent_logger.log_response(len(response), response_elapsed)
            log_ctx.record_step_time("FallbackResponse", response_elapsed)

            state = AgentStateManager.set_final_response(state, response)

        # 记录耗时汇总
        log_timing_summary(log_ctx)

        return state

    async def _think(
        self,
        state: AgentState,
        question: str,
        trace: Optional[Any] = None,
    ) -> tuple[str, Optional[str], Optional[Dict]]:
        """
        Generate a thought and decide on an action.

        Returns:
            Tuple of (thought, action_name, action_input).
        """
        # Build observations summary
        observations = ""
        for thought in state["thoughts"]:
            if hasattr(thought, 'observation') and thought.observation:
                observations += f"- {thought.observation}\n"
            elif isinstance(thought, dict) and thought.get("observation"):
                observations += f"- {thought['observation']}\n"

        # Get tool descriptions
        tool_descriptions = self._get_tool_descriptions()

        # Build prompt
        system_prompt = AGENT_SYSTEM_PROMPT.format(tools=tool_descriptions)
        thought_prompt = THOUGHT_PROMPT.format(
            question=question,
            observations=observations or "暂无",
        )

        messages = [
            {"role": "system", "content": system_prompt},
            *state["messages"],
            {"role": "user", "content": thought_prompt},
        ]

        # Get LLM response with optional tracing
        if trace:
            with trace.generation(
                name="think_llm_call",
                model=getattr(self.llm, 'model', 'unknown'),
                input={"prompt": thought_prompt[:500], "observations": observations[:500]},
                metadata={"phase": "thinking"},
            ) as gen:
                response = await self.llm.ainvoke(messages)
                gen.end(output=response[:1000])
        else:
            response = await self.llm.ainvoke(messages)

        # Parse the response
        parsed_thought, action, action_input = self._parse_thought_response(response)

        return parsed_thought, action, action_input

    async def _generate_response(self, state: AgentState, question: str) -> str:
        """Generate the final response based on reasoning."""
        # Build thoughts summary
        thoughts_text = ""
        for i, thought in enumerate(state["thoughts"], 1):
            if hasattr(thought, 'thought'):
                thoughts_text += f"{i}. {thought.thought}\n"
            elif isinstance(thought, dict):
                thoughts_text += f"{i}. {thought.get('thought', '')}\n"

        # Build observations summary
        observations_text = ""
        for tc in state["tool_calls"]:
            if hasattr(tc, 'name'):
                name = tc.name
                result = tc.result if tc.result else tc.error
            else:
                name = tc.get("name", "unknown")
                result = tc.get("result") or tc.get("error", "")
            observations_text += f"- {name}: {str(result)[:500]}\n"

        prompt = FINAL_RESPONSE_PROMPT.format(
            question=question,
            thoughts=thoughts_text or "直接回答",
            observations=observations_text or "无工具调用",
        )

        messages = [
            {"role": "system", "content": "你是一个专业的客服助手，请基于以下信息生成回答。"},
            {"role": "user", "content": prompt},
        ]

        response = await self.llm.ainvoke(messages)
        return response

    def _build_initial_messages(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]],
        conversation_id: Optional[str],
    ) -> List[Dict[str, str]]:
        """Build initial message list."""
        messages = []

        # Add memory context if available
        if conversation_id:
            context = self.memory.get_context(conversation_id)
            messages.extend(context)

        # Add provided history
        if history:
            messages.extend(history)

        # Add current question
        messages.append({"role": "user", "content": question})

        return messages

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
