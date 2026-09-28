/**
 * HTTP 客户端。
 *
 * 用 fetch 而不是 axios：SSE over POST 必须靠 fetch + ReadableStream，
 * axios 在浏览器里做不了流式响应，留着它只会变成第二套错误处理。
 */
import { ApiError, networkError, type ApiErrorPayload } from './errors'
import { loadIdentity } from './tenant'

const BASE = '/api'

function headers(extra?: HeadersInit): Headers {
  const h = new Headers(extra)
  const { tenantId, userId } = loadIdentity()
  // 🔴 每个请求都要带：后端 fail-closed，缺了一律 400
  if (tenantId) h.set('X-Tenant-Id', tenantId)
  if (userId) h.set('X-User-Id', userId)
  if (!h.has('Accept-Language')) h.set('Accept-Language', 'zh-CN')
  return h
}

async function toApiError(res: Response): Promise<ApiError> {
  let payload: ApiErrorPayload
  try {
    const body = await res.json()
    payload =
      typeof body === 'object' && body !== null && 'code' in body
        ? (body as ApiErrorPayload)
        : { code: 'INTERNAL', message: `请求失败（HTTP ${res.status}）` }
  } catch {
    payload = { code: 'INTERNAL', message: `请求失败（HTTP ${res.status}）` }
  }
  return new ApiError(payload, res.status)
}

interface RequestOptions {
  method?: string
  body?: unknown
  signal?: AbortSignal
  query?: Record<string, string | number | undefined>
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, signal, query } = options
  const url = new URL(`${BASE}${path}`, window.location.origin)
  for (const [k, v] of Object.entries(query ?? {})) {
    if (v !== undefined) url.searchParams.set(k, String(v))
  }

  let res: Response
  try {
    res = await fetch(url, {
      method,
      signal,
      headers: headers(body !== undefined ? { 'Content-Type': 'application/json' } : undefined),
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
  } catch (cause) {
    // 🔴 取消要原样抛出，不能包装成网络错误——上层靠它区分"用户主动停止"与"真失败"
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause
    throw networkError(cause)
  }

  if (!res.ok) throw await toApiError(res)
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export async function upload<T>(path: string, file: File, query?: Record<string, string>): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.origin)
  for (const [k, v] of Object.entries(query ?? {})) url.searchParams.set(k, v)

  const form = new FormData()
  form.append('file', file)

  let res: Response
  try {
    // 不设 Content-Type：浏览器要自己带 multipart boundary
    res = await fetch(url, { method: 'POST', headers: headers(), body: form })
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause
    throw networkError(cause)
  }
  if (!res.ok) throw await toApiError(res)
  return (await res.json()) as T
}

/**
 * SSE over POST。
 *
 * EventSource 只支持 GET 且不能带自定义头（拿不到 X-Tenant-Id），
 * 所以必须手动读 ReadableStream。
 */
export async function* streamEvents<T>(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): AsyncGenerator<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, {
      method: 'POST',
      signal,
      headers: headers({ 'Content-Type': 'application/json', Accept: 'text/event-stream' }),
      body: JSON.stringify(body),
    })
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause
    throw networkError(cause)
  }

  if (!res.ok) throw await toApiError(res)
  if (!res.body) throw networkError(new Error('响应没有可读流'))

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // SSE 以空行分隔事件；网络分片可能把一个事件切断，所以要留住残段
      const chunks = buffer.split('\n\n')
      buffer = chunks.pop() ?? ''
      for (const chunk of chunks) {
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data:')) continue
          const raw = line.slice(5).trim()
          if (!raw) continue
          try {
            yield JSON.parse(raw) as T
          } catch {
            /* 半个 JSON：丢弃该事件而不是让整条流挂掉 */
          }
        }
      }
    }
  } finally {
    // 🔴 无论正常结束还是被取消都要放掉 reader，否则连接泄漏、后端继续烧 token
    reader.cancel().catch(() => {})
  }
}
