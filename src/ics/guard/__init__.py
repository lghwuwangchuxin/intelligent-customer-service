"""L3-d guard：护栏。

🔴 与 rag / tools / memory **平级且互不 import**。
🔴 护栏**优先用编排框架的内置中间件**（PII / 人工确认 / 调用上限 / 重试 / 摘要）——
   它们会被编译成真实图节点，可视化、可调试、可检查点，自研版本通常缺这三样。
   这里只放框架没有的部分：预算、租户授权、写操作待确认单。
"""
from .authz import before_tool_call
from .budget import check_budget

__all__ = ["before_tool_call", "check_budget"]
