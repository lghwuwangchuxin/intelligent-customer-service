/** 后端响应形状的镜像。字段名与 src/ics/api/routes/*.py 一一对应。 */

// ── 对话 ─────────────────────────────────────────────────
export interface AgentInfo {
  id: string
  display_name: string
  description: string
  pattern: 'simple' | 'react' | 'plan_execute'
  tools: string[]
  requires_approval: string[]
}

export interface ToolCall {
  name: string | null
  args: unknown
}

export interface ChatReply {
  content: string
  agent_id: string
  conversation_id: string
  tool_calls: ToolCall[]
  interrupted: boolean
}

/** 流式事件。与后端 routes/chat.py 的 emit() 一一对应。 */
export type StreamEvent =
  | { type: 'agent'; agent_id: string; conversation_id: string }
  | { type: 'token'; content: string }
  | { type: 'node'; node: string }
  | { type: 'done' }

export interface ConversationSummary {
  conversation_id: string
  title: string
  agent_id: string
  message_count: number
  updated_at: string
}

export interface ConversationDetail {
  conversation_id: string
  messages: { role: string; content: string }[]
}

// ── 知识库 ───────────────────────────────────────────────
export interface RetrievedDoc {
  id: string
  doc_id: string
  chunk_no: number
  content: string
  score: number | null
  section: string | null
}

export interface RetrieveResult {
  query: string
  /** dense / hybrid / hybrid+rerank —— 能看出词法路是否降级了 */
  strategy: string
  latency_ms: number
  documents: RetrievedDoc[]
}

export interface DocumentInfo {
  doc_id: string
  kb_id: string
  title: string
  chunk_count: number
  updated_at: string
}

export interface IngestResult {
  doc_id: string
  chunks: number
  skipped_unchanged?: boolean
  format?: string
}

export interface KnowledgeStats {
  documents: number
  chunks: number
  /** 🔴 召回率基线。HNSW 退化完全无声，这是唯一能看出来的地方。 */
  last_recall: { recall: number; at: string } | null
}

// ── 运维 ─────────────────────────────────────────────────
export interface PendingAction {
  id: string
  tool: string
  args: Record<string, unknown>
  summary: string
  status: string
  created_at: string
}

export interface MemoryItem {
  key: string
  value: Record<string, unknown>
}

export interface SystemInfo {
  env: string
  models: Record<string, string>
  agents: string[]
  prompts: string[]
  tools: string[]
  /** false = 会话记忆降级为进程内，重启即丢、多副本不共享 */
  durable_memory: boolean
  retrieval: Record<string, unknown>
}

export interface Readiness {
  status: string
  database?: string
  durable_memory?: boolean
}
