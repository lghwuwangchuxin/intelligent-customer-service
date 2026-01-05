"""
Energy Analytics Tool.
Provides energy efficiency analysis and reporting.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.mock_data import EnergyMockData

logger = logging.getLogger(__name__)


class EnergyAnalyticsTool:
    """
    Tool for energy data analysis.

    Generates energy efficiency reports and usage statistics.
    """

    def __init__(self):
        """Initialize energy analytics tool."""
        self.mock_data = EnergyMockData()

    async def generate_report(
        self,
        station_id: str = "default_station",
        period: str = "month",
    ) -> Dict[str, Any]:
        """
        Generate energy efficiency report.

        Args:
            station_id: Station identifier
            period: Report period (day, week, month)

        Returns:
            Energy efficiency report
        """
        logger.info(f"Generating {period} report for station: {station_id}")
        return self.mock_data.generate_energy_report(station_id, period)

    def format_report(self, report: Dict[str, Any]) -> str:
        """Format energy report for display."""
        comparison = report.get("comparison_with_previous", {})

        lines = [
            f"**能效报告** - {report['report_period']}",
            f"生成日期: {report['report_date']}",
            f"",
            f"**能源使用概况**:",
            f"  总用电量: {report['total_energy_kwh']:,.2f} kWh",
            f"  能效率: {report['efficiency_percent']}%",
            f"  利用率: {report['utilization_percent']}%",
            f"",
            f"**负荷分析**:",
            f"  峰值负荷: {report['peak_load_kw']} kW",
            f"  平均负荷: {report['average_load_kw']} kW",
            f"  最低负荷: {report['min_load_kw']} kW",
            f"",
            f"**运营数据**:",
            f"  充电次数: {report['total_sessions']} 次",
            f"  平均时长: {report['avg_session_duration_min']} 分钟",
            f"  营收: {report['revenue']:,.2f} 元",
            f"  成本: {report['cost']:,.2f} 元",
            f"",
            f"**环保贡献**:",
            f"  减少碳排放: {report['carbon_saved_kg']:,.2f} kg",
            f"",
            f"**同比变化**:",
            f"  用电量: {comparison.get('energy_change_percent', 0):+.1f}%",
            f"  能效: {comparison.get('efficiency_change_percent', 0):+.1f}%",
            f"  利用率: {comparison.get('utilization_change_percent', 0):+.1f}%",
        ]
        return "\n".join(lines)

    async def get_efficiency_trend(
        self,
        station_id: str,
        periods: int = 7,
    ) -> List[Dict[str, Any]]:
        """Get efficiency trend data."""
        trends = []
        for i in range(periods):
            report = self.mock_data.generate_energy_report(station_id, "day")
            report["day"] = i + 1
            trends.append(report)
        return trends
