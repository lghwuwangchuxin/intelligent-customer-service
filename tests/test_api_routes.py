"""L5 api：路由层契约。

守的是「两条路径行为不一致」这类 bug —— 非流式对了、流式漏了，
而界面**只用流式**，所以单测覆盖非流式会给出虚假的安全感。
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from ics.api.routes import chat

SRC = Path(__file__).resolve().parents[1] / "src" / "ics"


def _calls_in(func_name: str) -> set[str]:
    """取出某个路由函数体内调用到的函数名（含嵌套定义）。"""
    tree = ast.parse(inspect.getsource(chat))
    target = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == func_name
    )
    return {
        n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
        for n in ast.walk(target)
        if isinstance(n, ast.Call)
    }


def test_both_chat_paths_index_the_conversation():
    """🔴 实跑抓到的真实 bug。

    流式端点原本**完全没有**更新会话索引，导致从界面发起的会话永远不进侧栏：
    实测 checkpointer 里 6 条 thread，索引表只有 1 条，且那条是 curl 打非流式接口写的。
    界面只走流式，所以这个 bug 在只测非流式时完全看不见。
    """
    assert "_index_conversation" in _calls_in("chat"), "非流式路径未更新会话索引"
    assert "_index_conversation" in _calls_in("chat_stream"), "流式路径未更新会话索引"


def test_both_chat_paths_scope_thread_by_tenant():
    """thread_id 必须带租户前缀，否则不同租户撞上同一 conversation_id 会串会话。"""
    for fn in ("chat", "chat_stream"):
        assert "_thread_id" in _calls_in(fn), f"{fn} 未使用带租户前缀的 thread_id"


def test_thread_id_includes_tenant():
    assert chat._thread_id("acme", "c1") == "acme:c1"
    assert chat._thread_id("acme", "c1") != chat._thread_id("other", "c1")


def test_index_failure_does_not_break_chat():
    """索引是附属动作：它失败不能让已经产出的回复丢掉。"""
    src = inspect.getsource(chat._index_conversation)
    assert "except Exception" in src, "索引失败未被兜住"
    assert "raise" in src.split("except asyncio.CancelledError:")[1].split("except")[0], (
        "取消必须重新抛出"
    )


def test_streaming_reraises_cancellation():
    """🔴 吞掉 CancelledError → 客户端断连后后台继续烧 token。"""
    src = inspect.getsource(chat.chat_stream)
    assert "asyncio.CancelledError" in src and "raise" in src


def test_every_route_requires_tenant():
    """所有业务路由都必须经过 Ctx 依赖（fail-closed 的入口）。"""
    tree = ast.parse((SRC / "api" / "routes" / "chat.py").read_text("utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        decorated = any(
            isinstance(d, ast.Call) and getattr(d.func, "attr", "") in
            {"get", "post", "delete", "put", "patch"}
            for d in node.decorator_list
        )
        if not decorated:
            continue
        args = {a.arg for a in node.args.args}
        assert "ctx" in args, f"路由 {node.name} 未要求租户上下文"
