"""
Cost Optimization Tool.
Provides cost analysis and optimization suggestions.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.mock_data import EnergyMockData

logger = logging.getLogger(__name__)


class CostOptimizationTool:
    """
    Tool for cost optimization analysis.

    Provides cost breakdown, savings opportunities, and recommendations.
    """

    def __init__(self):
        """Initialize cost optimization tool."""
        self.mock_data = EnergyMockData()

    async def analyze_and_optimize(
        self,
        station_id: str = "default_station",
    ) -> Dict[str, Any]:
        """
        Analyze costs and generate optimization suggestions.

        Args:
            station_id: Station identifier

        Returns:
            Optimization analysis and suggestions
        """
        logger.info(f"Analyzing costs for station: {station_id}")
        return self.mock_data.generate_optimization_suggestions(station_id)

    def format_optimization_report(self, report: Dict[str, Any]) -> str:
        """Format optimization report for display."""
        lines = [
            f"**成本优化分析报告**",
            f"分析日期: {report['analysis_date']}",
            f"",
            f"**当前成本概况**:",
            f"  月度运营成本: {report['current_monthly_cost']:,.2f} 元",
            f"  潜在节省空间: {report['potential_monthly_savings']:,.2f} 元",
            f"  节省比例: {report['savings_percentage']:.1f}%",
            f"",
            f"**峰谷时段**:",
            f"  峰时: {', '.join(report['peak_hours'])}",
            f"  谷时: {', '.join(report['valley_hours'])}",
            f"",
            f"**建议电价策略**:",
            f"  峰时电价: {report['recommended_pricing_strategy']['peak_price']:.2f} 元/kWh",
            f"  平时电价: {report['recommended_pricing_strategy']['normal_price']:.2f} 元/kWh",
            f"  谷时电价: {report['recommended_pricing_strategy']['valley_price']:.2f} 元/kWh",
            f"",
            f"**优化建议**:",
        ]

        for suggestion in report.get("suggestions", []):
            lines.extend([
                f"",
                f"  **{suggestion['id']}. {suggestion['title']}**",
                f"  {suggestion['description']}",
                f"  预计节省: {suggestion['potential_savings']:,.2f} 元/月",
                f"  实施难度: {suggestion['implementation_difficulty']}",
                f"  优先级: {suggestion['priority']}",
            ])

        return "\n".join(lines)

    def get_quick_wins(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get quick-win optimization opportunities."""
        suggestions = report.get("suggestions", [])
        return [
            s for s in suggestions
            if s["implementation_difficulty"] == "低" and s["priority"] == "高"
        ]
