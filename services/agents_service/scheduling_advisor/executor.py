"""
Scheduling Advisor Agent Executor.
Handles A2A requests for load forecasting and scheduling.
"""

import logging
import sys
from pathlib import Path

from a2a.server.agent_execution import RequestContext

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.base_executor import BaseAgentExecutor
from .tools.load_forecast import LoadForecastTool

logger = logging.getLogger(__name__)


class SchedulingAdvisorExecutor(BaseAgentExecutor):
    """
    Scheduling Advisor Agent Executor.

    Specializes in:
    - Load forecasting
    - Scheduling optimization
    - Peak-valley balancing
    """

    def __init__(self):
        """Initialize scheduling advisor executor."""
        super().__init__(
            agent_name="调度顾问",
            agent_description="预测用电负荷，优化调度计划",
        )
        self.forecast_tool = LoadForecastTool()

    def get_system_prompt(self) -> str:
        """Get system prompt for scheduling advisor."""
        return """你是一个专业的调度顾问，专门负责充电站的负荷预测和调度优化。

你的职责：
1. 预测未来用电负荷
2. 识别峰值和低谷时段
3. 提供调度优化建议
4. 帮助平衡电网负荷

关注指标：
- 负荷预测曲线
- 峰谷时段分布
- 设备调度策略
- 用户引导策略

回答要求：
- 预测要有置信度说明
- 建议要可执行
- 考虑电网约束
- 兼顾用户体验
"""

    async def process_request(
        self,
        user_message: str,
        context: RequestContext,
    ) -> str:
        """
        Process user request for load forecasting.

        Args:
            user_message: User's message
            context: Request context

        Returns:
            Response with forecast and scheduling advice
        """
        logger.info(f"Processing request: {user_message[:100]}...")

        # Determine forecast horizon
        hours = 24
        if any(kw in user_message for kw in ["今天", "12小时"]):
            hours = 12
        elif any(kw in user_message for kw in ["48小时", "两天"]):
            hours = 48

        # Generate forecast
        forecast = await self.forecast_tool.forecast(hours=hours)
        forecast_text = self.forecast_tool.format_forecast(forecast)

        # Get scheduling advice
        advice = self.forecast_tool.get_scheduling_advice(forecast)
        advice_text = "\n调度建议:\n" + "\n".join(f"- {a}" for a in advice)

        context_info = f"{forecast_text}\n{advice_text}"

        response = await self.call_llm(
            user_message=user_message,
            context_info=context_info,
            temperature=0.5,
        )

        return response
