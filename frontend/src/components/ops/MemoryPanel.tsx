import { useState } from 'react'
import { Brain } from 'lucide-react'
import { opsApi } from '../../api/ops'
import { useAsync } from '../../hooks/useAsync'
import { Empty, ErrorBox, Spinner } from '../ui/Primitives'

/**
 * 长期记忆（跨会话）。
 * 与会话历史是两层：会话历史是本次对话的图状态，这里是跨会话沉淀的用户偏好。
 */
export function MemoryPanel() {
  const [q, setQ] = useState('')
  const { data, error, loading, reload } = useAsync(() => opsApi.memories(q || undefined), [q])

  return (
    <div className="space-y-3">
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="搜索长期记忆…"
        className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-primary-500 focus:outline-none"
      />

      {loading && <Spinner />}
      {error && <ErrorBox error={error} onRetry={() => void reload()} />}
      {data?.length === 0 && <Empty>还没有沉淀的长期记忆</Empty>}

      {data?.map((m) => (
        <div key={m.key} className="rounded-lg border border-slate-200 bg-white p-3">
          <div className="mb-1.5 flex items-center gap-1.5">
            <Brain size={13} className="text-slate-400" />
            <span className="font-mono text-xs text-slate-700">{m.key}</span>
          </div>
          <pre className="overflow-x-auto text-xs text-slate-600">
            {JSON.stringify(m.value, null, 2)}
          </pre>
        </div>
      ))}
    </div>
  )
}
