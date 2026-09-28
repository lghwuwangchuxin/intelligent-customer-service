"""FastAPI 依赖：请求头 → RequestContext。"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from ..core.config import settings
from ..core.context import RequestContext
from ..core.errors import TenantRequired


def request_context(
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
    accept_language: Annotated[str | None, Header(alias="Accept-Language")] = None,
) -> RequestContext:
    """🔴 fail-closed：缺租户直接 400，绝不回落到默认租户。"""
    if not x_tenant_id:
        raise TenantRequired("缺少请求头 X-Tenant-Id")
    try:
        return RequestContext(
            tenant_id=x_tenant_id,
            user_id=x_user_id,
            locale=(accept_language or "zh-CN").split(",")[0],
            budget_tokens=settings.default_budget_tokens,
        )
    except ValueError as exc:
        raise TenantRequired(str(exc)) from exc


Ctx = Annotated[RequestContext, Depends(request_context)]


def container(request: Request) -> object:
    return request.app.state.container


Container = Annotated[object, Depends(container)]
