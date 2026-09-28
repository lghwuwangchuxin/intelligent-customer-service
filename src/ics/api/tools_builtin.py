"""内置工具的注册与 LangChain 适配。

放在 L5 而非 tools 层：工具**实现**依赖 rag（检索）等能力层，
而 tools 层与 rag 层平级、互不 import。组装发生在入口层是正确的方向。
"""
from __future__ import annotations

from typing import Annotated, Any

# 🔴 必须模块级导入：函数内导入时，`from __future__ import annotations` 会把
#    注解留成字符串，框架在解析 ToolRuntime[RequestContext] 时在模块全局里找不到它。
#    api 是 L5，在 agent 之上，导入编排框架不违反分层契约。
from langchain.tools import ToolRuntime, tool

from ..core.context import RequestContext
from ..core.telemetry import get_logger
from ..rag.generate.assemble import assemble
from ..rag.retrieve.pipeline import RetrievalPipeline
from ..tools.registry import ToolRegistry, ToolSpec

logger = get_logger(__name__)


def register_builtin_tools(registry: ToolRegistry, retrieval: RetrievalPipeline) -> None:
    async def knowledge_search(ctx: RequestContext, args: dict[str, Any]) -> str:
        result = await retrieval.retrieve(ctx, str(args["query"]), kb_id=args.get("kb_id"))
        return assemble(result.chunks)

    registry.register(ToolSpec(
        name="knowledge_search",
        description="在企业知识库中检索资料。回答事实性问题前应先调用它。",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "要检索的问题"}},
            "required": ["query"],
        },
        handler=knowledge_search,
    ))

    async def create_ticket(ctx: RequestContext, args: dict[str, Any]) -> str:
        # ⚠️ 接入点：应调用工单系统
        logger.info("创建工单 tenant=%s summary=%s", ctx.tenant_id, args.get("summary"))
        return f"[未接入工单系统] 已记录：{args.get('summary')}"

    registry.register(ToolSpec(
        name="create_ticket",
        description="当问题无法在对话中解决时，创建人工工单。",
        parameters={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "问题摘要"},
                "detail": {"type": "string", "description": "详情与已尝试的步骤"},
            },
            "required": ["summary"],
        },
        handler=create_ticket,
        is_write=True,
    ))

    async def query_order(ctx: RequestContext, args: dict[str, Any]) -> str:
        # ⚠️ 接入点：应调用订单服务
        return f"[未接入订单服务] 订单 {args.get('order_id')} 查询已记录"

    registry.register(ToolSpec(
        name="query_order",
        description="查询一笔充电订单的明细，包括电量、时长、费用构成与优惠。",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "充电订单号"}},
            "required": ["order_id"],
        },
        handler=query_order,
    ))

    async def issue_refund(ctx: RequestContext, args: dict[str, Any]) -> str:
        # ⚠️ 接入点：应调用支付服务。写操作，guard 层会转成待确认单。
        return f"[未接入支付服务] 退款请求已记录：{args.get('order_id')}"

    registry.register(ToolSpec(
        name="issue_refund",
        description="发起退款。敏感写操作，执行前会转为待人工确认的操作单。",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["order_id", "amount", "reason"],
        },
        handler=issue_refund,
        is_write=True,
    ))


def as_langchain_tools(registry: ToolRegistry, names: list[str]) -> list[Any]:
    """把注册表里的工具适配成 LangChain 工具。

    租户通过 ``ToolRuntime.context``（框架的类型化通道）显式下传，
    不出现在发给模型的参数里——否则模型可以自己编一个租户。
    """
    built: list[Any] = []
    for name in names:
        spec = registry.get(name)

        def make(spec: Any = spec) -> Any:
            @tool(spec.name, description=spec.description, parse_docstring=False)
            async def _fn(
                runtime: ToolRuntime[RequestContext],
                query: Annotated[str, "工具输入（JSON 或自然语言）"] = "",
                **kwargs: Any,
            ) -> str:
                args = dict(kwargs)
                if query:
                    args.setdefault("query", query)
                return await spec.handler(runtime.context, args)

            return _fn

        built.append(make())
    return built
