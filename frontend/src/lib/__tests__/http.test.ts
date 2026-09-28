import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from '../errors'
import { request, streamEvents } from '../http'

function sseResponse(chunks: string[]): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const enc = new TextEncoder()
      for (const c of chunks) controller.enqueue(enc.encode(c))
      controller.close()
    },
  })
  return new Response(stream, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

afterEach(() => vi.unstubAllGlobals())

describe('request', () => {
  it('🔴 每个请求都带 X-Tenant-Id —— 后端 fail-closed，缺了一律 400', async () => {
    localStorage.setItem('ics.tenant', 'acme')
    localStorage.setItem('ics.user', 'u1')
    // 显式给出 fetch 的签名，否则 mock 被推断成零参函数，取 calls[0][1] 通不过类型检查
    const fetchMock = vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>(
      async () => new Response('{}', { status: 200 }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await request('/chat/agents')

    const init = fetchMock.mock.calls[0]?.[1]
    const headers = init?.headers as Headers
    expect(headers.get('X-Tenant-Id')).toBe('acme')
    expect(headers.get('X-User-Id')).toBe('u1')
  })

  it('把后端错误契约还原成 ApiError', async () => {
    vi.stubGlobal('fetch', async () =>
      new Response(
        JSON.stringify({
          code: 'BUDGET_EXCEEDED',
          message: '本次请求已超出资源预算',
          detail: 'remaining=0',
          retryable: false,
        }),
        { status: 402 },
      ),
    )
    await expect(request('/chat')).rejects.toMatchObject({
      code: 'BUDGET_EXCEEDED',
      retryable: false,
      status: 402,
    })
  })

  it('非 JSON 错误体也要给出可用的 ApiError', async () => {
    vi.stubGlobal('fetch', async () => new Response('Internal Server Error', { status: 500 }))
    await expect(request('/chat')).rejects.toBeInstanceOf(ApiError)
  })

  it('🔴 取消要原样抛出，不能包装成网络错误', async () => {
    vi.stubGlobal('fetch', async () => {
      throw new DOMException('aborted', 'AbortError')
    })
    await expect(request('/chat')).rejects.toMatchObject({ name: 'AbortError' })
  })
})

describe('streamEvents', () => {
  it('解析 SSE 事件', async () => {
    vi.stubGlobal('fetch', async () =>
      sseResponse([
        'data: {"type":"agent","agent_id":"general"}\n\n',
        'data: {"type":"token","content":"你"}\n\n',
        'data: {"type":"done"}\n\n',
      ]),
    )
    const out = []
    for await (const ev of streamEvents('/chat/stream', {})) out.push(ev)
    expect(out).toEqual([
      { type: 'agent', agent_id: 'general' },
      { type: 'token', content: '你' },
      { type: 'done' },
    ])
  })

  it('🔴 事件被网络分片切断时必须能拼回来', async () => {
    // 真实网络里一个事件常被拆到两个 chunk，简单按 chunk 解析会丢数据
    vi.stubGlobal('fetch', async () =>
      sseResponse(['data: {"type":"tok', 'en","content":"好"}\n\n']),
    )
    const out = []
    for await (const ev of streamEvents('/chat/stream', {})) out.push(ev)
    expect(out).toEqual([{ type: 'token', content: '好' }])
  })

  it('半个 JSON 只丢该事件，不让整条流挂掉', async () => {
    vi.stubGlobal('fetch', async () =>
      sseResponse(['data: {坏的\n\n', 'data: {"type":"done"}\n\n']),
    )
    const out = []
    for await (const ev of streamEvents('/chat/stream', {})) out.push(ev)
    expect(out).toEqual([{ type: 'done' }])
  })

  it('HTTP 错误在开流前就抛 ApiError', async () => {
    vi.stubGlobal('fetch', async () =>
      new Response(JSON.stringify({ code: 'TENANT_REQUIRED', message: '缺少租户标识' }), {
        status: 400,
      }),
    )
    const gen = streamEvents('/chat/stream', {})
    await expect(gen.next()).rejects.toMatchObject({ code: 'TENANT_REQUIRED' })
  })
})
