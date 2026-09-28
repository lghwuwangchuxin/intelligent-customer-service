import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Bot, Brain, Building2, Cpu, Database, ShieldAlert } from 'lucide-react'
import { chatApi } from './api/chat'
import { opsApi } from './api/ops'
import { ChatInput } from './components/chat/ChatInput'
import { ChatMessage } from './components/chat/ChatMessage'
import { AgentPicker } from './components/chat/AgentPicker'
import { ConversationList } from './components/chat/ConversationList'
import { KnowledgePanel } from './components/knowledge/KnowledgePanel'
import { MemoryPanel } from './components/ops/MemoryPanel'
import { PendingPanel } from './components/ops/PendingPanel'
import { SystemPanel } from './components/ops/SystemPanel'
import { Badge, ErrorBox, Modal, Spinner, cx } from './components/ui/Primitives'
import { TenantGate } from './components/ui/TenantGate'
import { nodeLabel } from './lib/nodes'
import { useAsync } from './hooks/useAsync'
import { useChat } from './hooks/useChat'
import { useIdentity } from './hooks/useIdentity'

type Panel = 'knowledge' | 'pending' | 'memory' | 'system' | null

export default function App() {
  const { identity, update, ready } = useIdentity()

  // 🔴 没有租户就不发任何请求：后端 fail-closed，发出去只会得到一屏 400
  if (!ready) return <TenantGate identity={identity} onSubmit={update} />

  return <Workspace key={identity.tenantId} onSwitchTenant={() => update({ tenantId: '', userId: '' })} />
}

function Workspace({ onSwitchTenant }: { onSwitchTenant: () => void }) {
  const chat = useChat()
  const agents = useAsync(() => chatApi.agents(), [])
  const conversations = useAsync(() => chatApi.conversations(), [])
  const pending = useAsync(() => opsApi.pending(), [])

  const [agentId, setAgentId] = useState<string | null>(null)
  const [panel, setPanel] = useState<Panel>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chat.messages])

  const agentName = useMemo(() => {
    const byId = new Map(agents.data?.map((a) => [a.id, a.display_name || a.id]))
    return (id?: string) => (id ? byId.get(id) : undefined)
  }, [agents.data])

  const send = useCallback(
    async (text: string) => {
      await chat.send(text, agentId)
      // 会话列表与待确认清单都可能因这次对话变化
      void conversations.reload()
      void pending.reload()
    },
    [chat, agentId, conversations, pending],
  )

  const pendingCount = pending.data?.length ?? 0

  return (
    <div className="flex h-screen bg-slate-50">
      {/* 侧栏 */}
      <aside className="flex w-72 shrink-0 flex-col border-r border-slate-200 bg-white p-3">
        <div className="mb-3 flex items-center gap-2 px-1">
          <Bot className="text-primary-600" size={20} />
          <span className="font-semibold text-slate-800">智能客服</span>
        </div>

        <div className="mb-3 border-b border-slate-100 pb-3">
          <ConversationList
            items={conversations.data ?? []}
            activeId={chat.conversationId}
            onNew={chat.reset}
            onOpen={(id) => void chat.loadConversation(id)}
            onDelete={async (id) => {
              await chatApi.deleteConversation(id)
              if (id === chat.conversationId) chat.reset()
              void conversations.reload()
            }}
          />
        </div>

        <div className="mb-3 min-h-0 flex-1 overflow-y-auto">
          <p className="mb-1.5 px-1 text-xs font-medium text-slate-500">智能体</p>
          {agents.loading && <Spinner />}
          {agents.error && <ErrorBox error={agents.error} onRetry={() => void agents.reload()} />}
          {agents.data && (
            <AgentPicker agents={agents.data} value={agentId} onChange={setAgentId} />
          )}
        </div>

        <nav className="space-y-0.5 border-t border-slate-100 pt-2">
          <NavItem icon={<Database size={15} />} onClick={() => setPanel('knowledge')}>
            知识库
          </NavItem>
          <NavItem
            icon={<ShieldAlert size={15} />}
            onClick={() => setPanel('pending')}
            badge={pendingCount > 0 ? pendingCount : undefined}
          >
            待确认操作
          </NavItem>
          <NavItem icon={<Brain size={15} />} onClick={() => setPanel('memory')}>
            长期记忆
          </NavItem>
          <NavItem icon={<Cpu size={15} />} onClick={() => setPanel('system')}>
            系统信息
          </NavItem>
          <NavItem icon={<Building2 size={15} />} onClick={onSwitchTenant}>
            切换租户
          </NavItem>
        </nav>
      </aside>

      {/* 主区 */}
      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="mx-auto max-w-3xl space-y-5">
            {chat.messages.length === 0 && (
              <div className="pt-16 text-center">
                <Bot className="mx-auto mb-3 text-slate-300" size={40} />
                <p className="text-slate-500">
                  {agentId ? `正在使用「${agentName(agentId)}」` : '提问吧，服务端会自动分派智能体'}
                </p>
              </div>
            )}

            {chat.messages.map((m) => (
              <ChatMessage key={m.id} message={m} agentName={agentName(m.agentId)} />
            ))}

            {chat.activeNode && (
              <div className="flex items-center gap-2 pl-11 text-xs text-slate-400">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-primary-400" />
                <span title={chat.activeNode}>{nodeLabel(chat.activeNode)}</span>
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        </div>

        <ChatInput busy={chat.busy} onSend={(t) => void send(t)} onStop={chat.stop} />
      </main>

      {/* 面板 */}
      {panel === 'knowledge' && (
        <Modal title="知识库" wide onClose={() => setPanel(null)}>
          <KnowledgePanel />
        </Modal>
      )}
      {panel === 'pending' && (
        <Modal title="待确认操作" onClose={() => { setPanel(null); void pending.reload() }}>
          <PendingPanel />
        </Modal>
      )}
      {panel === 'memory' && (
        <Modal title="长期记忆" onClose={() => setPanel(null)}>
          <MemoryPanel />
        </Modal>
      )}
      {panel === 'system' && (
        <Modal title="系统信息" onClose={() => setPanel(null)}>
          <SystemPanel />
        </Modal>
      )}
    </div>
  )
}

function NavItem({
  icon,
  children,
  onClick,
  badge,
}: {
  icon: React.ReactNode
  children: React.ReactNode
  onClick: () => void
  badge?: number
}) {
  return (
    <button
      onClick={onClick}
      className={cx(
        'flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-sm text-slate-600',
        'hover:bg-slate-100',
      )}
    >
      <span className="text-slate-400">{icon}</span>
      <span className="flex-1 text-left">{children}</span>
      {badge !== undefined && <Badge tone="amber">{badge}</Badge>}
    </button>
  )
}
