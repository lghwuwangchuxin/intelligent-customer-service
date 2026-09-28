"""L0 core：内容模型 / 上下文 / 错误。"""
import pytest

from ics.core.context import RequestContext
from ics.core.errors import AppError, BudgetExceeded, ModelRateLimited
from ics.core.types import Content, ImagePart, LLMResponse, TextPart, Usage

# ── 内容模型 ────────────────────────────────────────────────

def test_text_only_sends_plain_string():
    """纯文本回字符串：厂商对它的兼容性最好。"""
    assert Content.text("user", "你好").to_openai() == {"role": "user", "content": "你好"}


def test_multimodal_becomes_parts():
    c = Content(role="user", parts=[TextPart(text="看图"), ImagePart(url="https://x/a.png")])
    blocks = c.to_openai()["content"]
    assert isinstance(blocks, list)
    assert blocks[1]["type"] == "image_url"


def test_discriminated_union_rejects_unknown_type():
    """带 discriminator 的联合类型能在解析期挡住脏数据。"""
    with pytest.raises(Exception):
        Content.model_validate({"role": "user", "parts": [{"type": "bogus"}]})


def test_as_text_ignores_non_text_parts():
    c = Content(role="user", parts=[TextPart(text="a"), ImagePart(url="u"), TextPart(text="b")])
    assert c.as_text() == "ab"


# ── usage ──────────────────────────────────────────────────

def test_reasoning_and_cached_tokens_are_separate():
    """🔴 推理 token 与缓存命中 token 必须单列，合并统计成本必错。"""
    u = Usage(prompt_tokens=1572, completion_tokens=200, reasoning_tokens=150, cached_tokens=1408)
    assert u.billable_input == 164     # 缓存命中的部分计价可低至 1/50
    assert u.total_tokens == 1772


def test_truncated_empty_detects_reasoning_budget_exhaustion():
    """🔴 推理模型把预算耗在思维链上：content 空 + finish_reason=length。"""
    assert LLMResponse(content="", finish_reason="length").truncated_empty
    assert not LLMResponse(content="有内容", finish_reason="length").truncated_empty
    assert not LLMResponse(content="", finish_reason="stop").truncated_empty


# ── 上下文 ──────────────────────────────────────────────────

def test_tenant_is_fail_closed():
    """🔴 缺租户即抛，绝不给默认值——默认值等于放行跨租户读取。"""
    with pytest.raises(ValueError):
        RequestContext(tenant_id="")


@pytest.mark.parametrize("bad", ["a b", "a/../b", "-x", "x" * 65, "a;drop"])
def test_malformed_tenant_rejected(bad):
    with pytest.raises(ValueError):
        RequestContext(tenant_id=bad)


def test_context_is_immutable():
    """不可变才能安全跨线程/协程传递。"""
    import dataclasses

    ctx = RequestContext(tenant_id="acme")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.tenant_id = "hacked"


def test_spend_returns_new_context():
    ctx = RequestContext(tenant_id="acme", budget_tokens=100)
    after = ctx.spend(30)
    assert ctx.remaining == 100 and after.remaining == 70


def test_child_splits_budget_to_prevent_multi_agent_runaway():
    """🔴 子 Agent 按份额切预算，否则多智能体会各跑各的、总花销失控。"""
    ctx = RequestContext(tenant_id="acme", budget_tokens=100, spent_tokens=40)
    child = ctx.child(share=3)
    assert child.budget_tokens == 20 and child.spent_tokens == 0
    assert child.trace_id == ctx.trace_id       # trace 继承，便于串起链路


def test_assert_owns_blocks_cross_tenant_row():
    ctx = RequestContext(tenant_id="acme")
    ctx.assert_owns("acme")
    with pytest.raises(PermissionError):
        ctx.assert_owns("other")


async def test_context_survives_task_and_thread_boundaries():
    """🔴 显式传参跨 Task/线程池后仍正确——这正是 ContextVar 会失败的地方。"""
    import asyncio

    ctx = RequestContext(tenant_id="acme")

    async def work(c):
        await asyncio.sleep(0)
        return await asyncio.to_thread(lambda: c.tenant_id)

    assert set(await asyncio.gather(*(work(ctx) for _ in range(20)))) == {"acme"}


# ── 错误 ────────────────────────────────────────────────────

def test_payload_never_leaks_internal_message():
    """🔴 响应体只出 code + 本地化文案 + 调用方 detail。

    原始异常 message 常夹带 prompt 片段与内部主机名。
    """
    payload = BudgetExceeded("remaining=0").to_payload()
    assert payload["code"] == "BUDGET_EXCEEDED"
    assert payload["message"] == "本次请求已超出资源预算"
    assert payload["detail"] == "remaining=0"


def test_locale_switches_message():
    assert BudgetExceeded().to_payload("en-US")["message"].startswith("Request exceeded")


def test_unknown_locale_falls_back_to_default_not_empty():
    assert BudgetExceeded().to_payload("xx-YY")["message"]


def test_retryable_is_first_class():
    """重试策略直接读 retryable，不靠猜异常类型。"""
    assert ModelRateLimited().retryable is True
    assert BudgetExceeded().retryable is False


def test_every_error_code_has_all_locale_messages():
    """新增错误码时漏翻译，外语用户会看到兜底文案——这条能在 CI 抓住。"""
    from ics.core.errors import MESSAGES

    def walk(cls):
        yield cls
        for sub in cls.__subclasses__():
            yield from walk(sub)

    codes = {c.code for c in walk(AppError)}
    for locale, table in MESSAGES.items():
        missing = codes - set(table)
        assert not missing, f"{locale} 缺少文案: {sorted(missing)}"
