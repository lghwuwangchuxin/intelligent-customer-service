import { request, streamEvents } from '../lib/http'
import type {
  AgentInfo,
  ChatReply,
  ConversationDetail,
  ConversationSummary,
  StreamEvent,
} from './types'

export interface ChatRequest {
  message: string
  agent_id?: string | null
  conversation_id?: string | null
  kb_id?: string
}

export const chatApi = {
  agents: () => request<{ agents: AgentInfo[] }>('/chat/agents').then((r) => r.agents),

  send: (body: ChatRequest, signal?: AbortSignal) =>
    request<ChatReply>('/chat', { method: 'POST', body, signal }),

  stream: (body: ChatRequest, signal?: AbortSignal) =>
    streamEvents<StreamEvent>('/chat/stream', body, signal),

  conversations: () =>
    request<{ conversations: ConversationSummary[] }>('/chat/conversations').then(
      (r) => r.conversations,
    ),

  conversation: (id: string) => request<ConversationDetail>(`/chat/conversations/${id}`),

  deleteConversation: (id: string) =>
    request<{ deleted: boolean }>(`/chat/conversations/${id}`, { method: 'DELETE' }),
}
