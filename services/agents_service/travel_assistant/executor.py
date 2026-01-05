"""
Travel Assistant Agent Executor.
Handles A2A requests for charging station search and navigation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from a2a.server.agent_execution import RequestContext

# Add parent directory to path for importing common module
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.base_executor import BaseAgentExecutor
from .tools.station_search import StationSearchTool

logger = logging.getLogger(__name__)


class TravelAssistantExecutor(BaseAgentExecutor):
    """
    Travel Assistant Agent Executor.

    Specializes in:
    - Searching nearby charging stations
    - Providing navigation suggestions
    - Recommending stations based on user preferences
    """

    def __init__(self):
        """Initialize travel assistant executor."""
        super().__init__(
            agent_name="出行助手",
            agent_description="帮助用户查找附近充电站，提供导航和推荐建议",
        )
        self.station_tool = StationSearchTool()

    def get_system_prompt(self) -> str:
        """Get system prompt for travel assistant."""
        return """你是一个专业的出行助手，专门帮助电动汽车车主查找充电站。

你的职责：
1. 根据用户位置搜索附近的充电站
2. 根据用户需求推荐合适的充电站（快充/慢充、价格、可用性等）
3. 提供充电站的详细信息，包括距离、价格、可用桩位、功率等
4. 给出导航建议

回答要求：
- 使用友好、专业的语气
- 信息要简洁明了
- 重要信息用数字和列表呈现
- 如果没有找到合适的充电站，提供替代建议

你可以使用以下工具：
- search_nearby: 搜索附近充电站
"""

    async def process_request(
        self,
        user_message: str,
        context: RequestContext,
    ) -> str:
        """
        Process user request for charging station search.

        Args:
            user_message: User's message
            context: Request context

        Returns:
            Response with charging station information
        """
        logger.info(f"Processing request: {user_message[:100]}...")

        # Parse user intent
        fast_charging = any(
            kw in user_message for kw in ["快充", "快速", "急", "赶时间"]
        )
        need_available = any(
            kw in user_message for kw in ["有空位", "有桩", "能用的"]
        )

        # Search for stations
        stations = await self.station_tool.search_nearby(
            radius_km=5.0,
            limit=5,
            fast_charging_only=fast_charging,
            min_available_ports=1 if need_available else 0,
        )

        # Format station info as context
        station_info = self.station_tool.format_stations_list(stations)

        # Generate response using LLM
        context_info = f"搜索结果：\n{station_info}"
        response = await self.call_llm(
            user_message=user_message,
            context_info=context_info,
            temperature=0.7,
        )

        return response
