"""
Fault Diagnosis Tool.
Provides fault identification and troubleshooting solutions.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.mock_data import ChargingMockData

logger = logging.getLogger(__name__)


class FaultDiagnosisTool:
    """
    Tool for diagnosing charging faults.

    Provides fault identification, root cause analysis, and solutions.
    """

    def __init__(self):
        """Initialize fault diagnosis tool."""
        self.mock_data = ChargingMockData()

        # Common fault keywords mapping to error codes
        self.fault_keywords = {
            "拔不出": "E001",
            "锁住": "E001",
            "解锁": "E001",
            "网络": "E002",
            "通信": "E002",
            "连接": "E002",
            "过热": "E003",
            "温度": "E003",
            "电压": "E004",
            "中断": "E005",
            "停止": "E005",
            "支付": "E006",
            "付款": "E006",
            "离线": "E007",
            "功率": "E008",
        }

    async def diagnose(
        self,
        error_code: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Diagnose a charging fault.

        Args:
            error_code: Optional error code (e.g., "E001")
            description: Optional fault description

        Returns:
            Fault diagnosis and solution
        """
        # If no error code, try to infer from description
        if not error_code and description:
            error_code = self._infer_error_code(description)

        if not error_code:
            error_code = "E000"  # Unknown error

        logger.info(f"Diagnosing fault: {error_code}")
        return self.mock_data.get_fault_solution(error_code)

    def _infer_error_code(self, description: str) -> Optional[str]:
        """Infer error code from description keywords."""
        for keyword, code in self.fault_keywords.items():
            if keyword in description:
                return code
        return None

    def get_all_error_codes(self) -> List[Dict[str, str]]:
        """Get list of all known error codes and names."""
        return [
            {"code": code, "name": info["name"]}
            for code, info in self.mock_data.ERROR_CODES.items()
        ]

    def format_diagnosis(self, diagnosis: Dict[str, Any]) -> str:
        """Format diagnosis result for display."""
        lines = [
            f"**故障诊断结果**",
            f"",
            f"**错误代码**: {diagnosis['error_code']}",
            f"**故障类型**: {diagnosis['error_name']}",
            f"**严重程度**: {diagnosis.get('severity', '未知')}",
            f"",
            f"**解决方案**:",
            f"{diagnosis['solution']}",
            f"",
            f"**预计处理时间**: {diagnosis.get('estimated_fix_time', '未知')}",
        ]

        if diagnosis.get("need_support"):
            lines.extend([
                f"",
                f"**需要人工支持**",
                f"客服电话: {diagnosis.get('support_phone', '400-xxx-xxxx')}",
            ])

        return "\n".join(lines)

    async def get_emergency_steps(self, fault_type: str) -> List[str]:
        """Get emergency handling steps for common faults."""
        emergency_steps = {
            "充电枪锁定": [
                "1. 不要强行拔出充电枪，以免损坏",
                "2. 长按充电枪上的解锁按钮3秒",
                '3. 如果仍无法解锁，在APP中点击"紧急解锁"',
                "4. 等待10秒后再次尝试拔出",
                "5. 如果问题持续，请联系客服",
            ],
            "充电中断": [
                "1. 检查车辆是否正常（仪表盘有无报警）",
                "2. 检查充电枪连接是否牢固",
                "3. 查看充电桩显示屏的错误信息",
                "4. 尝试重新插拔充电枪",
                "5. 如果问题持续，更换其他充电桩",
            ],
            "支付失败": [
                "1. 检查账户余额是否充足",
                "2. 确认支付方式是否有效",
                "3. 尝试更换支付方式",
                "4. 检查网络连接",
                "5. 如果问题持续，联系客服处理",
            ],
        }

        return emergency_steps.get(fault_type, [
            "1. 记录错误代码和现象",
            "2. 拍照保存现场情况",
            "3. 联系客服获取帮助",
        ])
