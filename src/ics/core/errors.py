"""错误体系。``retryable`` 是一等公民 —— 重试策略直接读它。"""
from __future__ import annotations

# 错误码 → 各语言文案。响应体只出这里的文案，不出原始异常 message。
MESSAGES: dict[str, dict[str, str]] = {
    "zh-CN": {
        "INTERNAL": "服务内部错误",
        "TENANT_REQUIRED": "缺少租户标识",
        "TENANT_MISMATCH": "数据归属与当前租户不符",
        "MODEL_RATE_LIMITED": "模型调用超出速率限制，请稍后重试",
        "MODEL_UNAVAILABLE": "模型服务暂时不可用",
        "CONTEXT_TOO_LONG": "输入超出模型上下文长度",
        "BUDGET_EXCEEDED": "本次请求已超出资源预算",
        "GUARDRAIL_BLOCKED": "该操作被安全策略拦截",
        "PENDING_APPROVAL": "该操作需要人工确认",
        "EMPTY_GENERATION": "模型未返回有效内容",
        "RETRIEVAL_FAILED": "知识检索失败",
        "UNSUPPORTED_DOCUMENT": "不支持的文档格式",
        "TOOL_FAILED": "工具执行失败",
        "AGENT_NOT_FOUND": "指定的智能体不存在",
        "UPSTREAM_UNAVAILABLE": "下游服务不可用",
        "IDEMPOTENCY_UNAVAILABLE": "幂等服务不可用，已拒绝执行",
    },
    "en-US": {
        "INTERNAL": "Internal server error",
        "TENANT_REQUIRED": "Tenant identifier is required",
        "TENANT_MISMATCH": "Record does not belong to the current tenant",
        "MODEL_RATE_LIMITED": "Model rate limit exceeded, please retry later",
        "MODEL_UNAVAILABLE": "Model service is temporarily unavailable",
        "CONTEXT_TOO_LONG": "Input exceeds the model context window",
        "BUDGET_EXCEEDED": "Request exceeded its resource budget",
        "GUARDRAIL_BLOCKED": "Blocked by safety policy",
        "PENDING_APPROVAL": "This action requires human approval",
        "EMPTY_GENERATION": "Model returned no usable content",
        "RETRIEVAL_FAILED": "Knowledge retrieval failed",
        "UNSUPPORTED_DOCUMENT": "Unsupported document format",
        "TOOL_FAILED": "Tool execution failed",
        "AGENT_NOT_FOUND": "Agent not found",
        "UPSTREAM_UNAVAILABLE": "Upstream service unavailable",
        "IDEMPOTENCY_UNAVAILABLE": "Idempotency backend unavailable; execution refused",
    },
}


class AppError(Exception):
    code: str = "INTERNAL"
    retryable: bool = False
    http_status: int = 500

    def __init__(self, detail: str = "", **ctx: object) -> None:
        self.detail = detail
        self.ctx = ctx
        super().__init__(detail or self.code)

    def to_payload(self, locale: str = "zh-CN") -> dict[str, object]:
        """🔴 只出 code + 本地化文案 + 调用方给的 detail。

        绝不回显原始异常 message —— 它常夹带 prompt 片段与内部主机名。
        """
        table = MESSAGES.get(locale) or MESSAGES["zh-CN"]
        payload: dict[str, object] = {
            "code": self.code,
            "message": table.get(self.code, table["INTERNAL"]),
            "detail": self.detail,
            "retryable": self.retryable,
        }
        if self.ctx:
            payload["context"] = self.ctx
        return payload


class TenantRequired(AppError):
    code, retryable, http_status = "TENANT_REQUIRED", False, 400


class TenantMismatch(AppError):
    code, retryable, http_status = "TENANT_MISMATCH", False, 403


class ModelRateLimited(AppError):
    code, retryable, http_status = "MODEL_RATE_LIMITED", True, 429


class ModelUnavailable(AppError):
    code, retryable, http_status = "MODEL_UNAVAILABLE", True, 503


class ContextTooLong(AppError):
    code, retryable, http_status = "CONTEXT_TOO_LONG", False, 400


class BudgetExceeded(AppError):
    code, retryable, http_status = "BUDGET_EXCEEDED", False, 402


class GuardrailBlocked(AppError):
    code, retryable, http_status = "GUARDRAIL_BLOCKED", False, 400


class PendingApproval(AppError):
    code, retryable, http_status = "PENDING_APPROVAL", False, 202


class EmptyGeneration(AppError):
    code, retryable, http_status = "EMPTY_GENERATION", True, 502


class RetrievalFailed(AppError):
    code, retryable, http_status = "RETRIEVAL_FAILED", True, 502


class UnsupportedDocument(AppError):
    code, retryable, http_status = "UNSUPPORTED_DOCUMENT", False, 415


class ToolFailed(AppError):
    code, retryable, http_status = "TOOL_FAILED", False, 502


class AgentNotFound(AppError):
    code, retryable, http_status = "AGENT_NOT_FOUND", False, 404


class UpstreamUnavailable(AppError):
    code, retryable, http_status = "UPSTREAM_UNAVAILABLE", True, 503


class IdempotencyUnavailable(AppError):
    code, retryable, http_status = "IDEMPOTENCY_UNAVAILABLE", True, 503
