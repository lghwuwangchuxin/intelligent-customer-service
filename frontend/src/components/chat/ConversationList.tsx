import { MessageSquare, Plus, Trash2 } from 'lucide-react'
import type { ConversationSummary } from '../../api/types'
import { cx } from '../ui/Primitives'

export function ConversationList({
  items,
  activeId,
  onOpen,
  onDelete,
  onNew,
}: {
  items: ConversationSummary[]
  activeId: string | null
  onOpen: (id: string) => void
  onDelete: (id: string) => void
  onNew: () => void
}) {
  return (
    <div className="flex h-full flex-col">
      <button
        onClick={onNew}
        className="mb-3 flex items-center justify-center gap-1.5 rounded-lg border border-slate-300 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
      >
        <Plus size={15} /> 新会话
      </button>

      <div className="flex-1 space-y-1 overflow-y-auto">
        {items.length === 0 && (
          <p className="py-6 text-center text-xs text-slate-400">还没有会话</p>
        )}
        {items.map((c) => (
          <div
            key={c.conversation_id}
            className={cx(
              'group flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm',
              activeId === c.conversation_id ? 'bg-primary-50' : 'hover:bg-slate-100',
            )}
          >
            <button
              onClick={() => onOpen(c.conversation_id)}
              className="flex min-w-0 flex-1 items-center gap-2 text-left"
            >
              <MessageSquare size={14} className="shrink-0 text-slate-400" />
              <span className="truncate text-slate-700">{c.title || '未命名会话'}</span>
            </button>
            <button
              onClick={() => onDelete(c.conversation_id)}
              title="删除会话"
              className="shrink-0 text-slate-300 opacity-0 transition hover:text-red-500 group-hover:opacity-100"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}
