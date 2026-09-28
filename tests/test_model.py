"""L2 model：别名表 / 网关记账与降级 / 嵌入陷阱。"""
import asyncio
from types import SimpleNamespace

import httpx
import pytest
import respx

from ics.core.context import RequestContext
from ics.core.errors import (
    BudgetExceeded,
    ContextTooLong,
    EmptyGeneration,
    ModelRateLimited,
    ModelUnavailable,
)
from ics.model.embedding import OllamaEmbedder, estimate_tokens
from ics.model.gateway import ModelGateway, _extract_usage
from ics.model.registry import ModelRegistry, ModelSpec

CTX = RequestContext(tenant_id="acme", budget_tokens=10_000)


# ── 别名表 ──────────────────────────────────────────────────

def test_upper_layers_use_aliases_only():
    """🔴 上层只认别名。写死厂商模型名会让换厂商变成全仓库改动。"""
    r = ModelRegistry()
    assert "chat-main" in r.aliases("chat")
    assert r.resolve("chat-main").provider_model


def test_unknown_alias_is_loud():
    with pytest.raises(ModelUnavailable):
        ModelRegistry().resolve("nope")


def test_embed_spec_exposes_limits_for_assembly_time_checks():
    """🔴 Embedder 暴露 max_input_tokens，让"块超限"在装配期就能校验。"""
    spec = ModelRegistry().resolve("embed-main")
    assert spec.dimensions > 0 and spec.max_input_tokens > 0


# ── 网关 ────────────────────────────────────────────────────

def _resp(content, *, model, finish="stop", usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None), finish_reason=finish
        )],
        model=model, usage=usage,
    )


@pytest.fixture
def gw():
    reg = ModelRegistry({
        "chat-main": ModelSpec("chat-main", "deepseek/deepseek-chat", "chat", 65536, 2048),
    })
    return ModelGateway(registry=reg, router=SimpleNamespace())


def _install(gw, resp):
    async def acompletion(**_kw):
        return resp
    gw._router = SimpleNamespace(acompletion=acompletion)


async def test_alias_redirect_is_flagged(gw, caplog):
    """🔴 请求 deepseek-chat 实际跑 deepseek-flash —— 不告警连计价都是错的。"""
    _install(gw, _resp("hi", model="deepseek-flash"))
    with caplog.at_level("WARNING"):
        out = await gw.chat(CTX, "chat-main", [])
    assert out.model == "deepseek-flash"
    assert any("重定向" in r.message for r in caplog.records)


async def test_no_warning_when_model_matches(gw, caplog):
    _install(gw, _resp("hi", model="deepseek-chat"))
    with caplog.at_level("WARNING"):
        await gw.chat(CTX, "chat-main", [])
    assert not any("重定向" in r.message for r in caplog.records)


async def test_empty_content_with_length_raises(gw):
    """🔴 推理模型耗光预算时不能静默返回空串。"""
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=30,
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=30))
    _install(gw, _resp("", model="deepseek-chat", finish="length", usage=usage))
    with pytest.raises(EmptyGeneration):
        await gw.chat(CTX, "chat-main", [])


async def test_budget_checked_before_spending(gw):
    """🔴 预算在发请求之前检查：已经超了就不该再花钱。"""
    _install(gw, _resp("hi", model="deepseek-chat"))
    broke = RequestContext(tenant_id="acme", budget_tokens=10, spent_tokens=10)
    with pytest.raises(BudgetExceeded):
        await gw.chat(broke, "chat-main", [])


async def test_rate_limit_maps_to_retryable_error(gw):
    async def boom(**_kw):
        raise RuntimeError("429 rate limit exceeded")
    gw._router = SimpleNamespace(acompletion=boom)
    with pytest.raises(ModelRateLimited) as exc:
        await gw.chat(CTX, "chat-main", [])
    assert exc.value.retryable is True


async def test_cancellation_propagates(gw):
    """🔴 吞掉 CancelledError → 任务泄漏 + token 白烧。"""
    async def cancel(**_kw):
        raise asyncio.CancelledError()
    gw._router = SimpleNamespace(acompletion=cancel)
    with pytest.raises(asyncio.CancelledError):
        await gw.chat(CTX, "chat-main", [])


def test_usage_separates_reasoning_and_cache():
    raw = SimpleNamespace(prompt_tokens=1572, completion_tokens=200,
                          completion_tokens_details=SimpleNamespace(reasoning_tokens=150),
                          prompt_cache_hit_tokens=1408)
    u = _extract_usage(SimpleNamespace(usage=raw))
    assert u.reasoning_tokens == 150 and u.cached_tokens == 1408


def test_usage_handles_dict_payload_and_missing():
    assert _extract_usage(SimpleNamespace(usage={"prompt_tokens": 5})).prompt_tokens == 5
    assert _extract_usage(SimpleNamespace(usage=None)).total_tokens == 0


# ── 嵌入 ────────────────────────────────────────────────────

@pytest.fixture
def embedder():
    reg = ModelRegistry({
        "embed-main": ModelSpec("embed-main", "bge-m3", "embed", 8192, 0,
                                dimensions=4, max_input_tokens=50),
    })
    return OllamaEmbedder(registry=reg, client=httpx.AsyncClient(base_url="http://ollama.test"))


@respx.mock
async def test_uses_embed_not_deprecated_endpoint(embedder):
    """🔴 /api/embeddings 已弃用，必须打 /api/embed。"""
    route = respx.post("http://ollama.test/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3, 0.4]]})
    )
    await embedder.embed(CTX, ["你好"])
    assert route.called


@respx.mock
async def test_truncate_disabled(embedder):
    """🔴 Ollama truncate 默认 true，会静默砍掉超长输入。"""
    import json as _json
    seen = {}

    def handler(request):
        seen.update(_json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3, 0.4]]})

    respx.post("http://ollama.test/api/embed").mock(side_effect=handler)
    await embedder.embed(CTX, ["你好"])
    assert seen["truncate"] is False


@respx.mock
async def test_dimension_mismatch_is_loud(embedder):
    """🔴 换嵌入模型没重建索引是本领域最危险的变更。"""
    respx.post("http://ollama.test/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
    )
    with pytest.raises(ModelUnavailable, match="维度"):
        await embedder.embed(CTX, ["你好"])


@respx.mock
async def test_count_mismatch_detected(embedder):
    """请求 2 条只回 1 条时必须报错，否则向量与切块错位对齐。"""
    respx.post("http://ollama.test/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3, 0.4]]})
    )
    with pytest.raises(ModelUnavailable, match="条数"):
        await embedder.embed(CTX, ["a", "b"])


async def test_overlong_raises_instead_of_truncating(embedder):
    with pytest.raises(ContextTooLong):
        await embedder.embed(CTX, ["超长" * 200])


@respx.mock
async def test_query_kind_adds_instruction_only_for_asymmetric_models():
    """🔴 非对称模型查询侧要加指令前缀；bge-m3 不需要，不该乱加。"""
    import json as _json
    reg = ModelRegistry({
        "embed-main": ModelSpec("embed-main", "bge-large-zh-v1.5", "embed", 512, 0,
                                dimensions=4, max_input_tokens=512),
    })
    e = OllamaEmbedder(registry=reg, client=httpx.AsyncClient(base_url="http://ollama.test"))
    seen = []

    def handler(request):
        seen.append(_json.loads(request.content)["input"][0])
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3, 0.4]]})

    respx.post("http://ollama.test/api/embed").mock(side_effect=handler)
    await e.embed(CTX, ["退款"], kind="query")
    await e.embed(CTX, ["退款"], kind="document")
    assert seen[0].startswith("为这个句子生成表示")
    assert seen[1] == "退款"


@respx.mock
async def test_cancellation_propagates_in_embed(embedder):
    async def cancel(_req):
        raise asyncio.CancelledError()
    respx.post("http://ollama.test/api/embed").mock(side_effect=cancel)
    with pytest.raises(asyncio.CancelledError):
        await embedder.embed(CTX, ["你好"])


def test_token_estimate():
    assert estimate_tokens("中文") == 2
    assert estimate_tokens("abcd") == 1
