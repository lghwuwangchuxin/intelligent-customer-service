"""
Load Forecast Tool.
Provides load prediction and scheduling optimization.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.mock_data import EnergyMockData

logger = logging.getLogger(__name__)


class LoadForecastTool:
    """
    Tool for load forecasting and scheduling.

    Provides load predictions and scheduling recommendations.
    """

    def __init__(self):
        """Initialize load forecast tool."""
        self.mock_data = EnergyMockData()

    async def forecast(
        self,
        station_id: str = "default_station",
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Generate load forecast.

        Args:
            station_id: Station identifier
            hours: Forecast horizon in hours

        Returns:
            Load forecast data
        """
        logger.info(f"Generating {hours}h forecast for station: {station_id}")
        return self.mock_data.generate_load_forecast(station_id, hours)

    def format_forecast(self, forecast: Dict[str, Any]) -> str:
        """Format load forecast for display."""
        summary = forecast.get("summary", {})

        lines = [
            f"**负荷预测报告**",
            f"生成时间: {forecast['forecast_generated_at']}",
            f"预测时长: {forecast['forecast_hours']} 小时",
            f"",
            f"**负荷概况**:",
            f"  最高预测负荷: {summary.get('max_predicted_load_kw', 0):.2f} kW",
            f"  平均预测负荷: {summary.get('avg_predicted_load_kw', 0):.2f} kW",
            f"  最低预测负荷: {summary.get('min_predicted_load_kw', 0):.2f} kW",
            f"  预计总用电量: {summary.get('total_predicted_energy_kwh', 0):.2f} kWh",
            f"",
            f"**峰值时段**: {summary.get('peak_hours', [])}",
            f"",
            f"**建议**:",
        ]

        for rec in forecast.get("recommendations", []):
            lines.append(f"  - {rec}")

        lines.append(f"\n**逐时预测 (前12小时)**:")
        for pred in forecast.get("predictions", [])[:12]:
            peak_mark = " [峰]" if pred.get("is_peak") else ""
            lines.append(
                f"  第{pred['hour']+1}小时: {pred['predicted_load_kw']:.1f} kW "
                f"(置信度: {pred['confidence']*100:.0f}%){peak_mark}"
            )

        return "\n".join(lines)

    def get_scheduling_advice(self, forecast: Dict[str, Any]) -> List[str]:
        """Get scheduling advice based on forecast."""
        advice = []
        summary = forecast.get("summary", {})
        peak_hours = summary.get("peak_hours", [])

        if peak_hours:
            advice.append(f"峰值时段为第 {peak_hours} 小时，建议启用功率调度")

        max_load = summary.get("max_predicted_load_kw", 0)
        if max_load > 300:
            advice.append("预测高负荷，建议提前检查设备状态")

        advice.append("建议在低谷时段安排设备维护")
        advice.append("可考虑引导用户错峰充电")

        return advice
