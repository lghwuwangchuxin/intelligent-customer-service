/**
 * 后端错误契约的前端镜像。
 *
 * 后端统一返回 { code, message, detail, retryable }：
 *   - message 已是本地化文案，直接展示，前端不要再翻译一遍
 *   - retryable 决定是否给「重试」按钮，不要靠猜 HTTP 状态码
 *   - 🔴 后端绝不回显原始异常 message（会夹带 prompt 片段与内部主机名），
 *     所以前端拿不到堆栈是**设计如此**，不要试图从别处补。
 */
export interface ApiErrorPayload {
  code: string
  message: string
  detail?: string
  retryable?: boolean
  context?: Record<string, unknown>
}

export class ApiError extends Error {
  readonly code: string
  readonly detail: string
  readonly retryable: boolean
  readonly status: number
  readonly context?: Record<string, unknown>

  constructor(payload: ApiErrorPayload, status: number) {
    super(payload.message || payload.code)
    this.name = 'ApiError'
    this.code = payload.code
    this.detail = payload.detail ?? ''
    this.retryable = payload.retryable ?? false
    this.status = status
    this.context = payload.context
  }

  /** 缺租户：UI 应引导去设置租户，而不是当成普通报错。 */
  get isTenantMissing(): boolean {
    return this.code === 'TENANT_REQUIRED'
  }

  /** 写操作被转成待确认单——这不是失败，是按设计挂起等人工批准。 */
  get isPendingApproval(): boolean {
    return this.code === 'PENDING_APPROVAL'
  }
}

/** 网络层失败（连不上、被取消之外的传输错误）没有后端契约，这里补一个。 */
export function networkError(cause: unknown): ApiError {
  return new ApiError(
    {
      code: 'NETWORK_ERROR',
      message: '无法连接到服务，请检查后端是否已启动',
      detail: cause instanceof Error ? cause.message : String(cause),
      retryable: true,
    },
    0,
  )
}
