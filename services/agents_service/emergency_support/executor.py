"""
Emergency Support Agent Executor.
Handles A2A requests for fault diagnosis and troubleshooting.
"""

import logging
import re
import sys
from pathlib import Path

from a2a.server.agent_execution import RequestContext

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.base_executor import BaseAgentExecutor
from .tools.fault_diagnosis import FaultDiagnosisTool

logger = logging.getLogger(__name__)


class EmergencySupportExecutor(BaseAgentExecutor):
    """
    Emergency Support Agent Executor.

    Specializes in:
    - Fault diagnosis
    - Troubleshooting guidance
    - Emergency support
    """

    def __init__(self):
        """Initialize emergency support executor."""
        super().__init__(
            agent_name="故障急救",
            agent_description="快速诊断充电故障，提供即时解决方案",
        )
        self.diagnosis_tool = FaultDiagnosisTool()

    def get_system_prompt(self) -> str:
        """Get system prompt for emergency support."""
        return """你是一个专业的故障急救专家，专门帮助用户解决充电过程中遇到的问题。

你的职责：
1. 快速识别故障类型
2. 提供清晰的解决步骤
3. 判断是否需要人工介入
4. 在紧急情况下提供安全建议

常见故障类型：
- E001: 充电枪锁定故障
- E002: 通信中断
- E003: 过温保护
- E004: 电压异常
- E005: 充电中断
- E006: 支付失败
- E007: 充电桩离线
- E008: 功率不匹配

回答要求：
- 语气要冷静、专业
- 步骤要清晰、可操作
- 涉及安全问题要特别提醒
- 如果无法远程解决，要指导用户联系客服
"""

    async def process_request(
        self,
        user_message: str,
        context: RequestContext,
    ) -> str:
        """
        Process user request for fault diagnosis.

        Args:
            user_message: User's message describing the fault
            context: Request context

        Returns:
            Response with diagnosis and solution
        """
        logger.info(f"Processing fault request: {user_message[:100]}...")

        # Try to extract error code from message
        error_code = self._extract_error_code(user_message)

        # Diagnose the fault
        diagnosis = await self.diagnosis_tool.diagnose(
            error_code=error_code,
            description=user_message,
        )

        # Format diagnosis
        diagnosis_info = self.diagnosis_tool.format_diagnosis(diagnosis)

        # Get emergency steps if applicable
        fault_type = diagnosis.get("error_name", "")
        steps = await self.diagnosis_tool.get_emergency_steps(fault_type)
        if steps:
            steps_text = "\n".join(steps)
            diagnosis_info += f"\n\n**紧急处理步骤**:\n{steps_text}"

        response = await self.call_llm(
            user_message=user_message,
            context_info=f"故障诊断结果：\n{diagnosis_info}",
            temperature=0.5,  # Lower temperature for more focused responses
        )

        return response

    def _extract_error_code(self, message: str) -> str | None:
        """Extract error code from user message."""
        # Match patterns like E001, E-001, 错误码001, etc.
        patterns = [
            r"E-?(\d{3})",
            r"错误码[：:]?\s*(\d{3})",
            r"故障码[：:]?\s*(\d{3})",
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return f"E{match.group(1)}"

        return None
