"""L3-d guard 与 L4 agent：护栏、写操作闸门、Agent 即数据。"""
import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from ics.agent.engine import AgentEngine
from ics.agent.loader import AgentStore
from ics.agent.patterns.router import route
from ics.agent.prompts import PromptLibrary
from ics.agent.schema import AgentConfig
from ics.core.context import RequestContext
from ics.core.errors import (
    AgentNotFound,
    BudgetExceeded,
    GuardrailBlocked,
    PendingApproval,
)
from ics.core.types import LLMResponse
from ics.guard import before_tool_call, check_budget
from ics.guard.pii import BUILTIN_TYPES, CUSTOM_DETECTORS, detector_for

CTX = RequestContext(tenant_id="acme")
AGENTS = AgentStore("agents")
PROMPTS = PromptLibrary("prompts")


# ── 预算 ────────────────────────────────────────────────────

def test_budget_blocks_when_exhausted():
    with pytest.raises(BudgetExceeded):
        check_budget(RequestContext(tenant_id="acme", budget_tokens=10, spent_tokens=10))


def test_budget_accounts_for_need():
    """预留量也要算：剩 5 但这次要花 100，应当提前拦住。"""
    with pytest.raises(BudgetExceeded):
        check_budget(RequestContext(tenant_id="acme", budget_tokens=100, spent_tokens=95), need=100)


# ── 工具闸门 ────────────────────────────────────────────────

class _Policy:
    def __init__(self, allow=True, write=()):
        self._allow, self._write = allow, set(write)

    def allowed(self, tenant_id, name):
        return self._allow

    def is_write(self, name):
        return name in self._write


class _Pending:
    async def create(self, ctx, tool, args, summary):
        return "p1"


async def test_read_tool_passes():
    await before_tool_call(CTX, "search", {}, _Policy())


async def test_disallowed_tool_blocked():
    with pytest.raises(GuardrailBlocked):
        await before_tool_call(CTX, "search", {}, _Policy(allow=False))


async def test_write_becomes_pending_action_not_execution():
    """🔴 写操作只产出待确认操作单，没有绕过路径。"""
    with pytest.raises(PendingApproval) as exc:
        await before_tool_call(CTX, "refund", {"amt": 1}, _Policy(write=["refund"]), _Pending())
    assert exc.value.ctx["pending_id"] == "p1"


async def test_write_without_pending_store_is_refused_not_executed():
    """没有操作单存储时必须拒绝，绝不能退化成直接执行。"""
    with pytest.raises(GuardrailBlocked):
        await before_tool_call(CTX, "refund", {}, _Policy(write=["refund"]))


# ── PII ─────────────────────────────────────────────────────

def test_chinese_pii_types_exist():
    """🔴 框架内置 PII 不含中国大陆手机号/身份证，而这是中文客服最常见的 PII。"""
    assert "cn_mobile" in CUSTOM_DETECTORS and "cn_id_card" in CUSTOM_DETECTORS
    assert "phone" not in BUILTIN_TYPES      # 内置里确实没有


def test_builtin_types_delegate_to_framework():
    assert detector_for("email") is None


def test_unknown_pii_type_is_loud():
    with pytest.raises(ValueError):
        detector_for("telepathy")


@pytest.mark.parametrize("text,hit", [
    ("我的号码是13812345678", True),
    ("订单号202513812345678999", False),     # 长数字串不应误伤
    ("座机 010-12345678", False),
])
def test_cn_mobile_regex(text, hit):
    import re

    assert bool(re.search(CUSTOM_DETECTORS["cn_mobile"], text)) is hit


# ── Agent 即数据 ────────────────────────────────────────────

def test_no_agent_class_exists_in_code():
    """🔴 代码里不存在任何一个具体 Agent —— 只有配置。"""
    ids = {a.id for a in AGENTS.list()}
    assert {"general", "billing", "escalation"} <= ids


def test_adding_an_agent_is_config_only(tmp_path):
    """加 Agent = 加一个 yaml，不改代码。"""
    (tmp_path / "newbie.yaml").write_text(
        "id: newbie\ndescription: 新来的\nprompt_ref: agent.general.v1\ntools: []\n",
        encoding="utf-8",
    )
    assert AgentStore(tmp_path).get("newbie").id == "newbie"


def test_unknown_agent_raises():
    with pytest.raises(AgentNotFound):
        AGENTS.get("nope")


def test_agents_reference_prompts_not_inline_text():
    """🔴 提示词放 prompts/，不进配置也不进代码。"""
    refs = set(PROMPTS.refs())
    for a in AGENTS.list():
        assert a.prompt_ref in refs, f"{a.id} 的 prompt_ref {a.prompt_ref} 不存在"


def test_agents_use_model_aliases_not_vendor_names():
    """配置里只能出现别名。"""
    from ics.model.registry import ModelRegistry

    valid = set(ModelRegistry().aliases())
    for a in AGENTS.list():
        assert a.model in valid, f"{a.id} 的 model={a.model} 不是已登记别名"


def test_agents_reference_registered_tools_only():
    from ics.api.tools_builtin import register_builtin_tools
    from ics.tools.registry import ToolRegistry

    reg = ToolRegistry()
    register_builtin_tools(reg, object())
    known = set(reg.names())
    for a in AGENTS.list():
        unknown = set(a.tools) - known
        assert not unknown, f"{a.id} 引用了未注册工具: {sorted(unknown)}"


def test_every_agent_has_step_limits():
    """🔴 ReAct 无收敛保证，必须限步。"""
    for a in AGENTS.list():
        assert a.guard.max_model_calls > 0 and a.guard.max_tool_calls > 0


def test_agent_holding_write_tool_declares_hitl():
    """持有写工具的 Agent 必须声明人工闸门，否则闸门形同虚设。"""
    for a in AGENTS.list():
        if "issue_refund" in a.tools:
            assert "issue_refund" in a.guard.hitl_tools


# ── 引擎 ────────────────────────────────────────────────────

@pytest.fixture
def engine():
    return AgentEngine(
        model_factory=lambda cfg: GenericFakeChatModel(messages=iter(["ok"] * 50)),
        tool_factory=lambda cfg: [],
        prompts=PROMPTS,
        checkpointer=__import__(
            "langgraph.checkpoint.memory", fromlist=["InMemorySaver"]
        ).InMemorySaver(),
    )


def test_middleware_compiles_into_real_graph_nodes(engine):
    """🔴 内置 middleware 编译成真实图节点——可视化、可调试、可检查点。"""
    nodes = set(engine.build(AGENTS.get("escalation")).get_graph().nodes)
    assert any(n.startswith("ModelCallLimitMiddleware") for n in nodes)
    assert any(n.startswith("ToolCallLimitMiddleware") for n in nodes)
    assert any(n.startswith("HumanInTheLoopMiddleware") for n in nodes)
    assert any("cn_mobile" in n for n in nodes), "中文手机号脱敏未进图"


def test_agents_without_write_tools_have_no_hitl_node(engine):
    """闸门只加在需要处，否则普通问答也会被中断。"""
    nodes = set(engine.build(AGENTS.get("general")).get_graph().nodes)
    assert not any(n.startswith("HumanInTheLoopMiddleware") for n in nodes)


def test_system_prompt_always_carries_boundary_rule(engine):
    for a in AGENTS.list():
        assert "<retrieved_documents>" in engine.system_prompt(a)


def test_compiled_graphs_are_cached_per_version(engine):
    cfg = AGENTS.get("general")
    assert engine.build(cfg) is engine.build(cfg)
    engine.invalidate(cfg)
    assert engine.build(cfg) is not None


# ── 路由 ────────────────────────────────────────────────────

class _LLM:
    def __init__(self, reply):
        self._reply = reply

    async def chat(self, ctx, model, messages, **kw):
        # BaseException 而非 Exception：CancelledError 不继承 Exception，
        # 用 Exception 判断会让取消被当成普通返回值——这正是生产代码里
        # `except Exception` 放行 CancelledError 的原因。
        if isinstance(self._reply, BaseException):
            raise self._reply
        return LLMResponse(content=self._reply)


async def test_router_picks_valid_agent():
    assert await route(CTX, "退款", AGENTS.list(), _LLM("billing"), PROMPTS) == "billing"


async def test_router_falls_back_on_unknown_reply():
    """路由返回未知 id 时回落，不让整个请求失败。"""
    assert await route(CTX, "x", AGENTS.list(), _LLM("不存在的"), PROMPTS) == "general"


async def test_router_falls_back_on_llm_failure():
    assert await route(CTX, "x", AGENTS.list(), _LLM(RuntimeError("down")), PROMPTS) == "general"


async def test_router_propagates_cancellation():
    with pytest.raises(asyncio.CancelledError):
        await route(CTX, "x", AGENTS.list(), _LLM(asyncio.CancelledError()), PROMPTS)


async def test_single_candidate_skips_llm_call():
    """只有一个候选时不该白花一次模型调用。"""
    one = [AgentConfig(id="only", prompt_ref="agent.general.v1")]
    assert await route(CTX, "x", one, _LLM(RuntimeError("不该被调用")), PROMPTS) == "only"
