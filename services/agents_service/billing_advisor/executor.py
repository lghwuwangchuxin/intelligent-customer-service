"""
Billing Advisor Agent Executor.
Handles A2A requests for billing queries and explanations.
"""

import logging
import sys
from pathlib import Path

from a2a.server.agent_execution import RequestContext

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.base_executor import BaseAgentExecutor
from .tools.billing_query import BillingQueryTool

logger = logging.getLogger(__name__)


class BillingAdvisorExecutor(BaseAgentExecutor):
    """
    Billing Advisor Agent Executor.

    Specializes in:
    - Billing history queries
    - Fee breakdown explanations
    - Consumption analysis
    """

    def __init__(self):
        """Initialize billing advisor executor."""
        super().__init__(
            agent_name="费用顾问",
            agent_description="查询充电账单，解释费用明细",
        )
        self.billing_tool = BillingQueryTool()

    def get_system_prompt(self) -> str:
        """Get system prompt for billing advisor."""
        return """你是一个专业的费用顾问，专门帮助用户理解充电费用。

你的职责：
1. 查询并展示用户的充电账单
2. 解释账单中的各项费用
3. 分析用户的充电消费习惯
4. 提供省钱建议

费用说明：
- 电费：按充电电量计算，单位为元/度
- 服务费：充电站收取的服务费用
- 停车费：部分站点收取的停车费用
- 折扣：优惠券或活动优惠

回答要求：
- 费用明细要清晰准确
- 对用户的疑问给予耐心解释
- 适当提供节省费用的建议
"""

    async def process_request(
        self,
        user_message: str,
        context: RequestContext,
    ) -> str:
        """
        Process user request for billing information.

        Args:
            user_message: User's message
            context: Request context

        Returns:
            Response with billing information
        """
        logger.info(f"Processing request: {user_message[:100]}...")

        # Determine if user wants summary or detailed history
        wants_summary = any(
            kw in user_message for kw in ["统计", "总共", "一共", "汇总", "分析"]
        )

        if wants_summary:
            summary = await self.billing_tool.get_summary()
            context_info = self.billing_tool.format_summary(summary)
        else:
            bills = await self.billing_tool.query_history(limit=5)
            context_info = self.billing_tool.format_bills_list(bills)

        response = await self.call_llm(
            user_message=user_message,
            context_info=f"账单信息：\n{context_info}",
            temperature=0.7,
        )

        return response
