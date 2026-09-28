"""配置 → 可执行图。

🔴 **换编排框架时只有这个文件和 patterns/ 需要重写**（爆炸半径表的承诺）。
   langchain / langgraph 的 import 只允许出现在 agent 层，由 .importlinter 强制。

🔴 护栏一律用内置 middleware：它们编译成**真实图节点**（节点名形如
   ``PIIMiddleware[email].before_model``），可视化、可调试、可检查点。
"""
from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)

from ..core.telemetry import get_logger
from ..guard.pii import detector_for
from .prompts import PromptLibrary, get_prompts
from .schema import AgentConfig

logger = get_logger(__name__)


class AgentEngine:
    """把 AgentConfig 编译成可执行图，并按 (id, version) 缓存。"""

    def __init__(
        self,
        model_factory: Any,
        tool_factory: Any,
        prompts: PromptLibrary | None = None,
        checkpointer: Any = None,
        store: Any = None,
    ) -> None:
        self._model_factory = model_factory
        self._tool_factory = tool_factory
        self._prompts = prompts or get_prompts()
        self._checkpointer = checkpointer
        self._store = store
        self._compiled: dict[tuple[str, int], Any] = {}

    def _middleware(self, cfg: AgentConfig, model: Any) -> list[Any]:
        mw: list[Any] = []

        # 进模型前脱敏。🔴 框架内置类型不含中国大陆手机号/身份证，
        # 这类规则由 guard.pii 提供正则，在这里装配。
        for pii in cfg.guard.pii_types:
            mw.append(PIIMiddleware(
                pii, detector=detector_for(pii), strategy="redact",
                apply_to_input=True, apply_to_output=True,
            ))

        # 🔴 ReAct 是贪心搜索，无收敛保证：工具持续返回空会无限循环烧钱
        mw.append(ModelCallLimitMiddleware(
            thread_limit=cfg.guard.max_model_calls * 3,
            run_limit=cfg.guard.max_model_calls, exit_behavior="end",
        ))
        mw.append(ToolCallLimitMiddleware(
            thread_limit=cfg.guard.max_tool_calls * 3,
            run_limit=cfg.guard.max_tool_calls, exit_behavior="end",
        ))
        # 工具瞬时失败重试（可靠性，区别于上面的次数上限）
        mw.append(ToolRetryMiddleware(max_retries=2, backoff_factor=2.0))
        # 长会话上下文自动摘要
        mw.append(SummarizationMiddleware(model=model, trigger={"tokens": 24_000}))

        # 人工闸门：interrupt 会把图挂起并落进 checkpointer，批准后原地续跑
        if cfg.guard.hitl_tools:
            mw.append(HumanInTheLoopMiddleware(
                interrupt_on={t: True for t in cfg.guard.hitl_tools},
                description_prefix="该操作需要人工审批",
            ))
        return mw

    def system_prompt(self, cfg: AgentConfig) -> str:
        from ..rag.generate.assemble import BOUNDARY_RULE

        base = self._prompts.load(cfg.prompt_ref) if cfg.prompt_ref else ""
        # 🔴 边界规则始终附加，防间接提示注入
        return f"{base.strip()}\n\n{BOUNDARY_RULE}"

    def build(self, cfg: AgentConfig) -> Any:
        if cfg.key in self._compiled:
            return self._compiled[cfg.key]
        model = self._model_factory(cfg)
        tools = self._tool_factory(cfg)
        graph = create_agent(
            model=model,
            tools=tools,
            system_prompt=self.system_prompt(cfg),
            middleware=self._middleware(cfg, model),
            checkpointer=self._checkpointer,
            store=self._store,
            name=cfg.id,
        )
        self._compiled[cfg.key] = graph
        logger.info(
            "Agent 已编译 id=%s v%d tools=%s hitl=%s",
            cfg.id, cfg.version, [getattr(t, "name", t) for t in tools], cfg.guard.hitl_tools,
        )
        return graph

    def invalidate(self, cfg: AgentConfig | None = None) -> None:
        if cfg is None:
            self._compiled.clear()
        else:
            self._compiled.pop(cfg.key, None)
