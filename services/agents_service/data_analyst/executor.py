"""
Data Analyst Agent Executor.
Handles A2A requests for energy data analysis.
"""

import logging
import sys
from pathlib import Path

from a2a.server.agent_execution import RequestContext

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.base_executor import BaseAgentExecutor
from .tools.energy_analytics import EnergyAnalyticsTool

logger = logging.getLogger(__name__)


class DataAnalystExecutor(BaseAgentExecutor):
    """
    Data Analyst Agent Executor.

    Specializes in:
    - Energy efficiency analysis
    - Usage statistics and trends
    - Performance reporting
    """

    def __init__(self):
        """Initialize data analyst executor."""
        super().__init__(
            agent_name="数据分析师",
            agent_description="分析能源使用数据，生成能效报告",
        )
        self.analytics_tool = EnergyAnalyticsTool()

    def get_system_prompt(self) -> str:
        """Get system prompt for data analyst."""
        return """你是一个专业的数据分析师，专门分析充电站的能源使用数据。

你的职责：
1. 分析能源使用效率
2. 生成能效报告
3. 识别用电趋势和异常
4. 提供数据驱动的洞察

分析维度：
- 总用电量和能效率
- 负荷分析（峰值、平均、最低）
- 利用率分析
- 营收和成本分析
- 环保贡献

回答要求：
- 数据要准确、有支撑
- 趋势分析要有对比
- 异常情况要重点标注
- 给出数据背后的业务含义
"""

    async def process_request(
        self,
        user_message: str,
        context: RequestContext,
    ) -> str:
        """
        Process user request for energy analysis.

        Args:
            user_message: User's message
            context: Request context

        Returns:
            Response with energy analysis
        """
        logger.info(f"Processing request: {user_message[:100]}...")

        # Determine report period
        period = "month"
        if any(kw in user_message for kw in ["今天", "日", "今日"]):
            period = "day"
        elif any(kw in user_message for kw in ["本周", "周", "这周"]):
            period = "week"

        # Generate report
        report = await self.analytics_tool.generate_report(period=period)
        report_text = self.analytics_tool.format_report(report)

        response = await self.call_llm(
            user_message=user_message,
            context_info=f"能效分析报告：\n{report_text}",
            temperature=0.5,
        )

        return response
