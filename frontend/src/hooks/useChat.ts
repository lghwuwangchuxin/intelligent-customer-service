import { useCallback, useRef, useState } from 'react'
import { chatApi } from '../api/chat'
import type { ToolCall } from '../api/types'
import { ApiError } from '../lib/errors'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  agentId?: string
  toolCalls?: ToolCall[]
  /** 被人工闸门挂起：不是失败，是等待批准 */
  interrupted?: boolean
  error?: ApiError
  streaming?: boolean
}

let seq = 0
const nextId = () => `m${++seq}`

/**
 * 对话。单 Agent 与多 Agent 是同一套机制：
 * 不传 agentId 就由后端路由分派，传了就直接用。
 */
export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [activeNode, setActiveNode] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const stop = useCallback(() => {
    // 🔴 abort 会让后端的流式生成器收到取消并真正结束，而不是继续烧 token
    abortRef.current?.abort()
    abortRef.current = null
  }, [])

  const reset = useCallback(() => {
    stop()
    setMessages([])
    setConversationId(null)
    setActiveNode(null)
  }, [stop])

  const loadConversation = useCallback(async (id: string) => {
    stop()
    const detail = await chatApi.conversation(id)
    setConversationId(id)
    setMessages(
      detail.messages
        .filter((m) => m.role === 'human' || m.role === 'ai')
        .map((m) => ({
          id: nextId(),
          role: m.role === 'human' ? ('user' as const) : ('assistant' as const),
          content: m.content,
        })),
    )
  }, [stop])

  const send = useCallback(
    async (text: string, agentId?: string | null, kbId = 'default') => {
      const trimmed = text.trim()
      if (!trimmed || busy) return

      const controller = new AbortController()
      abortRef.current = controller
      setBusy(true)
      setActiveNode(null)

      const replyId = nextId()
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: 'user', content: trimmed },
        { id: replyId, role: 'assistant', content: '', streaming: true },
      ])

      const patch = (fn: (m: Message) => Message) =>
        setMessages((prev) => prev.map((m) => (m.id === replyId ? fn(m) : m)))

      try {
        const events = chatApi.stream(
          {
            message: trimmed,
            agent_id: agentId ?? null,
            conversation_id: conversationId,
            kb_id: kbId,
          },
          controller.signal,
        )

        for await (const ev of events) {
          switch (ev.type) {
            case 'agent':
              setConversationId(ev.conversation_id)
              patch((m) => ({ ...m, agentId: ev.agent_id }))
              break
            case 'token':
              patch((m) => ({ ...m, content: m.content + ev.content }))
              break
            case 'node':
              // 图节点名（含 PIIMiddleware / HumanInTheLoop 等中间件节点），
              // 用于给用户一个"正在做什么"的提示
              setActiveNode(ev.node)
              break
            case 'done':
              patch((m) => ({ ...m, streaming: false }))
              break
          }
        }
        patch((m) => ({ ...m, streaming: false }))
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          // 用户主动停止：保留已生成的内容，不当成错误
          patch((m) => ({ ...m, streaming: false }))
        } else if (err instanceof ApiError) {
          patch((m) => ({
            ...m,
            streaming: false,
            interrupted: err.isPendingApproval,
            error: err.isPendingApproval ? undefined : err,
            content: err.isPendingApproval
              ? '该操作需要人工审批，已提交待确认清单。'
              : m.content,
          }))
        } else {
          throw err
        }
      } finally {
        setBusy(false)
        setActiveNode(null)
        abortRef.current = null
      }
    },
    [busy, conversationId],
  )

  return { messages, conversationId, busy, activeNode, send, stop, reset, loadConversation }
}
