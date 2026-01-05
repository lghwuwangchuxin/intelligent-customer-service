"""
Device Health Tool.
Provides device health monitoring and predictive maintenance.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.mock_data import EnergyMockData

logger = logging.getLogger(__name__)


class DeviceHealthTool:
    """
    Tool for device health monitoring.

    Provides health status, predictive maintenance, and alerts.
    """

    def __init__(self):
        """Initialize device health tool."""
        self.mock_data = EnergyMockData()

    async def check_health(
        self,
        device_id: str = "default_device",
    ) -> Dict[str, Any]:
        """
        Check device health status.

        Args:
            device_id: Device identifier

        Returns:
            Device health status
        """
        logger.info(f"Checking health for device: {device_id}")
        return self.mock_data.generate_device_health(device_id)

    async def check_multiple_devices(
        self,
        device_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Check health of multiple devices."""
        return [
            await self.check_health(device_id)
            for device_id in device_ids
        ]

    def format_health_report(self, health: Dict[str, Any]) -> str:
        """Format device health report."""
        lines = [
            f"**设备健康报告**",
            f"",
            f"**设备信息**:",
            f"  设备ID: {health['device_id']}",
            f"  设备类型: {health['device_type']}",
            f"",
            f"**健康状态**:",
            f"  健康评分: {health['health_score']}/100",
            f"  状态: {health['status']}",
            f"  运行时间: {health['uptime_percent']}%",
            f"  故障风险: {health['predicted_failure_risk']*100:.1f}%",
            f"",
            f"**运行统计**:",
            f"  累计充电次数: {health['total_charging_sessions']:,}",
            f"  累计充电量: {health['total_energy_delivered_kwh']:,} kWh",
            f"",
            f"**维护信息**:",
            f"  上次维护: {health['last_maintenance_date']}",
            f"  下次维护: {health['next_maintenance_date']}",
            f"",
            f"**组件健康**:",
        ]

        for component in health.get("components", []):
            lines.append(
                f"  - {component['name']}: {component['health']}/100 "
                f"(最后检查: {component['last_check']})"
            )

        if health.get("alerts"):
            lines.append(f"\n**告警信息**:")
            for alert in health["alerts"]:
                lines.append(f"  [{alert['type']}] {alert['message']}")

        return "\n".join(lines)

    def get_maintenance_recommendations(
        self,
        health: Dict[str, Any]
    ) -> List[str]:
        """Get maintenance recommendations based on health status."""
        recommendations = []

        if health["health_score"] < 70:
            recommendations.append("建议立即安排维护检查")

        if health["predicted_failure_risk"] > 0.2:
            recommendations.append("设备故障风险较高，建议预防性维护")

        for component in health.get("components", []):
            if component["health"] < 75:
                recommendations.append(
                    f"{component['name']}健康度较低，建议检查"
                )

        if not recommendations:
            recommendations.append("设备运行正常，按计划维护即可")

        return recommendations
