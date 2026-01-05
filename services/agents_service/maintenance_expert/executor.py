"""
Maintenance Expert Agent Executor.
Handles A2A requests for device health monitoring.
"""

import logging
import sys
from pathlib import Path

from a2a.server.agent_execution import RequestContext

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.base_executor import BaseAgentExecutor
from .tools.device_health import DeviceHealthTool

logger = logging.getLogger(__name__)


class MaintenanceExpertExecutor(BaseAgentExecutor):
    """
    Maintenance Expert Agent Executor.

    Specializes in:
    - Device health monitoring
    - Predictive maintenance
    - Maintenance scheduling
    """

    def __init__(self):
        """Initialize maintenance expert executor."""
        super().__init__(
            agent_name="运维专家",
            agent_description="监测设备健康状态，预测维护需求",
        )
        self.health_tool = DeviceHealthTool()

    def get_system_prompt(self) -> str:
        """Get system prompt for maintenance expert."""
        return """你是一个专业的运维专家，专门负责充电设备的健康监测和维护管理。

你的职责：
1. 监测设备健康状态
2. 分析设备运行数据
3. 预测潜在故障风险
4. 提供维护建议

关注指标：
- 设备健康评分
- 组件状态
- 运行时间和次数
- 故障风险预测

回答要求：
- 健康评估要有依据
- 风险提示要明确
- 维护建议要可操作
- 对紧急情况要重点标注
"""

    async def process_request(
        self,
        user_message: str,
        context: RequestContext,
    ) -> str:
        """
        Process user request for device health check.

        Args:
            user_message: User's message
            context: Request context

        Returns:
            Response with health assessment
        """
        logger.info(f"Processing request: {user_message[:100]}...")

        # Get device health
        health = await self.health_tool.check_health()
        health_report = self.health_tool.format_health_report(health)

        # Get recommendations
        recommendations = self.health_tool.get_maintenance_recommendations(health)
        recommendations_text = "\n".join(f"- {r}" for r in recommendations)

        context_info = (
            f"设备健康报告：\n{health_report}\n\n"
            f"维护建议：\n{recommendations_text}"
        )

        response = await self.call_llm(
            user_message=user_message,
            context_info=context_info,
            temperature=0.5,
        )

        return response
