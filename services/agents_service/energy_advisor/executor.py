"""
Energy Advisor Agent Executor.
Handles A2A requests for cost optimization.
"""

import logging
import sys
from pathlib import Path

from a2a.server.agent_execution import RequestContext

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.base_executor import BaseAgentExecutor
from .tools.cost_optimization import CostOptimizationTool

logger = logging.getLogger(__name__)


class EnergyAdvisorExecutor(BaseAgentExecutor):
    """
    Energy Advisor Agent Executor.

    Specializes in:
    - Cost analysis
    - Savings opportunities
    - Optimization recommendations
    """

    def __init__(self):
        """Initialize energy advisor executor."""
        super().__init__(
            agent_name="能源顾问",
            agent_description="分析能源成本，提供优化建议",
        )
        self.optimization_tool = CostOptimizationTool()

    def get_system_prompt(self) -> str:
        """Get system prompt for energy advisor."""
        return """你是一个专业的能源顾问，专门帮助充电站运营商优化能源成本。

你的职责：
1. 分析当前能源成本结构
2. 识别节省机会
3. 提供可操作的优化建议
4. 评估投资回报

优化方向：
- 峰谷电价优化
- 功率调度优化
- 储能系统配置
- 光伏发电接入
- 电价策略调整

回答要求：
- 数据分析要有依据
- 建议要具体可操作
- 考虑实施难度和优先级
- 给出预期的投资回报
"""

    async def process_request(
        self,
        user_message: str,
        context: RequestContext,
    ) -> str:
        """
        Process user request for cost optimization.

        Args:
            user_message: User's message
            context: Request context

        Returns:
            Response with optimization suggestions
        """
        logger.info(f"Processing request: {user_message[:100]}...")

        # Get optimization analysis
        report = await self.optimization_tool.analyze_and_optimize()
        report_text = self.optimization_tool.format_optimization_report(report)

        # Get quick wins
        quick_wins = self.optimization_tool.get_quick_wins(report)
        if quick_wins:
            quick_wins_text = "\n快速见效方案:\n" + "\n".join(
                f"- {w['title']}: 预计节省 {w['potential_savings']:.2f} 元/月"
                for w in quick_wins
            )
        else:
            quick_wins_text = ""

        context_info = f"{report_text}{quick_wins_text}"

        response = await self.call_llm(
            user_message=user_message,
            context_info=context_info,
            temperature=0.5,
        )

        return response
