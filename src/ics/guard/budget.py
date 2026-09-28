"""预算闸。

🔴 与 ReAct 步数上限是两回事：步数管循环次数，预算管花销。
   一次调用可以只走 3 步却塞进 200K token——步数不超、成本超了拦不住。
"""
from __future__ import annotations

from ..core.context import RequestContext
from ..core.errors import BudgetExceeded


def check_budget(ctx: RequestContext, need: int = 0) -> None:
    """调用**前**检查。已经超了就不要再发请求。"""
    if ctx.remaining <= need:
        raise BudgetExceeded(
            f"remaining={ctx.remaining}, need={need}", tenant=ctx.tenant_id
        )
