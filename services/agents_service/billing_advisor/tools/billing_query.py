"""
Billing Query Tool.
Provides billing history and expense analysis.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.mock_data import ChargingMockData

logger = logging.getLogger(__name__)


class BillingQueryTool:
    """
    Tool for querying billing information.

    Provides billing history, expense breakdown, and payment records.
    """

    def __init__(self):
        """Initialize billing query tool."""
        self.mock_data = ChargingMockData()

    async def query_history(
        self,
        user_id: str = "default_user",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Query billing history.

        Args:
            user_id: User identifier
            limit: Maximum number of records

        Returns:
            List of billing records
        """
        logger.info(f"Querying billing history for user: {user_id}, limit: {limit}")
        return self.mock_data.generate_billing_history(user_id, limit)

    async def get_summary(
        self,
        user_id: str = "default_user",
        limit: int = 30,
    ) -> Dict[str, Any]:
        """
        Get billing summary statistics.

        Args:
            user_id: User identifier
            limit: Number of records to analyze

        Returns:
            Summary statistics
        """
        bills = await self.query_history(user_id, limit)

        total_amount = sum(b["total_amount"] for b in bills)
        total_energy = sum(b["energy_kwh"] for b in bills)
        total_sessions = len(bills)

        return {
            "total_amount": round(total_amount, 2),
            "total_energy_kwh": round(total_energy, 2),
            "total_sessions": total_sessions,
            "avg_amount_per_session": round(total_amount / total_sessions, 2) if total_sessions else 0,
            "avg_energy_per_session": round(total_energy / total_sessions, 2) if total_sessions else 0,
            "avg_price_per_kwh": round(total_amount / total_energy, 2) if total_energy else 0,
        }

    def format_bill(self, bill: Dict[str, Any]) -> str:
        """Format a single bill record."""
        return (
            f"- **{bill['date']}** {bill['station_name']}\n"
            f"  充电量: {bill['energy_kwh']}kWh | "
            f"电费: {bill['electricity_cost']}元 | "
            f"服务费: {bill['service_fee']}元 | "
            f"合计: {bill['total_amount']}元"
        )

    def format_bills_list(self, bills: List[Dict[str, Any]]) -> str:
        """Format list of bills."""
        if not bills:
            return "暂无充电账单记录。"

        lines = [f"最近 {len(bills)} 条充电账单：\n"]
        for bill in bills:
            lines.append(self.format_bill(bill))

        return "\n".join(lines)

    def format_summary(self, summary: Dict[str, Any]) -> str:
        """Format billing summary."""
        return (
            f"**充电消费统计**\n\n"
            f"- 总消费金额: {summary['total_amount']} 元\n"
            f"- 总充电量: {summary['total_energy_kwh']} kWh\n"
            f"- 充电次数: {summary['total_sessions']} 次\n"
            f"- 平均每次消费: {summary['avg_amount_per_session']} 元\n"
            f"- 平均每次充电量: {summary['avg_energy_per_session']} kWh\n"
            f"- 平均单价: {summary['avg_price_per_kwh']} 元/kWh"
        )
