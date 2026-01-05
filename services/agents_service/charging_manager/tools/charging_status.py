"""
Charging Status Tool.
Provides real-time charging session monitoring.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.mock_data import ChargingMockData

logger = logging.getLogger(__name__)


class ChargingStatusTool:
    """
    Tool for monitoring charging sessions.

    Provides real-time status of ongoing charging sessions.
    """

    def __init__(self):
        """Initialize charging status tool."""
        self.mock_data = ChargingMockData()

    async def get_status(
        self,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get current charging session status.

        Args:
            session_id: Optional session ID (generates mock if not provided)

        Returns:
            Charging session status information
        """
        logger.info(f"Getting charging status for session: {session_id or 'current'}")
        return self.mock_data.generate_charging_status(session_id)

    def format_status(self, status: Dict[str, Any]) -> str:
        """
        Format charging status for display.

        Args:
            status: Status dictionary

        Returns:
            Formatted status string
        """
        lines = [
            f"**充电状态**: {status['status_cn']}",
            f"",
            f"**电量信息**:",
            f"  当前电量: {status['soc_percent']}%",
            f"  目标电量: {status['target_soc']}%",
            f"",
            f"**充电参数**:",
            f"  功率: {status['power_kw']} kW",
            f"  电压: {status['voltage_v']} V",
            f"  电流: {status['current_a']} A",
            f"",
            f"**充电统计**:",
            f"  已充电量: {status['energy_kwh']} kWh",
            f"  充电时长: {status['duration_minutes']} 分钟",
            f"  预计剩余: {status['estimated_minutes']} 分钟",
            f"  当前费用: {status['cost_current']} 元",
            f"",
            f"**充电站信息**:",
            f"  站点: {status['station_name']}",
            f"  桩号: {status['port_number']}",
            f"  开始时间: {status['start_time']}",
        ]
        return "\n".join(lines)
