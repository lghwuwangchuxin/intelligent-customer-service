"""
Charging Manager Agent Executor.
Handles A2A requests for charging status monitoring.
"""

import logging
import sys
from pathlib import Path

from a2a.server.agent_execution import RequestContext

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.base_executor import BaseAgentExecutor
from .tools.charging_status import ChargingStatusTool

logger = logging.getLogger(__name__)


class ChargingManagerExecutor(BaseAgentExecutor):
    """
    Charging Manager Agent Executor.

    Specializes in:
    - Real-time charging status monitoring
    - Charging time estimation
    - Power and cost tracking
    """

    def __init__(self):
        """Initialize charging manager executor."""
        super().__init__(
            agent_name="充电管家",
            agent_description="实时监控充电过程，提供充电状态和预估",
        )
        self.status_tool = ChargingStatusTool()

    def get_system_prompt(self) -> str:
        """Get system prompt for charging manager."""
        return """你是一个专业的充电管家，专门监控电动汽车的充电过程。

你的职责：
1. 提供实时充电状态信息
2. 估算充电完成时间
3. 监控充电功率和费用
4. 在异常情况下提供提醒

回答要求：
- 使用清晰、简洁的语言
- 重要数据用醒目的方式呈现
- 预估时间要考虑实际充电曲线
- 对异常情况给予温馨提示

你可以使用以下工具：
- get_status: 获取当前充电状态
"""

    async def process_request(
        self,
        user_message: str,
        context: RequestContext,
    ) -> str:
        """
        Process user request for charging status.

        Args:
            user_message: User's message
            context: Request context

        Returns:
            Response with charging status
        """
        logger.info(f"Processing request: {user_message[:100]}...")

        # Get current charging status
        status = await self.status_tool.get_status()

        # Format status as context
        status_info = self.status_tool.format_status(status)

        # Generate response using LLM
        context_info = f"当前充电状态：\n{status_info}"
        response = await self.call_llm(
            user_message=user_message,
            context_info=context_info,
            temperature=0.7,
        )

        return response
