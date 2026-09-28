import { useState } from 'react'
import { Check, ShieldAlert, X } from 'lucide-react'
import { opsApi } from '../../api/ops'
import { useAsync } from '../../hooks/useAsync'
import { Empty, ErrorBox, Spinner } from '../ui/Primitives'

/**
 * 待确认写操作。
 *
 * 这是新架构的核心安全机制在界面上的落点：**写操作不会被直接执行**，
 * 服务端把它转成一条操作单，必须有人在这里批准或拒绝。
 */
export function PendingPanel() {
  const { data, error, loading, reload } = useAsync(() => opsApi.pending(), [])
  const [working, setWorking] = useState<string | null>(null)

  const resolve = async (id: string, status: 'approved' | 'rejected') => {
    setWorking(id)
    try {
      await opsApi.resolve(id, status)
      await reload()
    } finally {
      setWorking(null)
    }
  }

  if (loading) return <Spinner />
  if (error) return <ErrorBox error={error} onRetry={() => void reload()} />
  if (!data?.length) return <Empty>没有待确认的操作</Empty>

  return (
    <div className="space-y-3">
      <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
        <ShieldAlert size={16} className="mt-0.5 shrink-0" />
        <span>以下操作会修改数据，服务端不会自动执行，需在此明确批准。</span>
      </div>

      {data.map((p) => (
        <div key={p.id} className="rounded-lg border border-slate-200 bg-white p-3">
          <div className="mb-2 flex items-center gap-2">
            <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs text-slate-700">
              {p.tool}
            </span>
            <span className="text-xs text-slate-400">
              {new Date(p.created_at).toLocaleString('zh-CN')}
            </span>
          </div>

          <p className="mb-2 text-sm text-slate-700">{p.summary}</p>

          <pre className="mb-3 overflow-x-auto rounded bg-slate-50 p-2 text-xs text-slate-600">
            {JSON.stringify(p.args, null, 2)}
          </pre>

          <div className="flex gap-2">
            <button
              disabled={working === p.id}
              onClick={() => void resolve(p.id, 'approved')}
              className="flex items-center gap-1 rounded-lg bg-green-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-green-700 disabled:bg-slate-300"
            >
              <Check size={13} /> 批准
            </button>
            <button
              disabled={working === p.id}
              onClick={() => void resolve(p.id, 'rejected')}
              className="flex items-center gap-1 rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              <X size={13} /> 拒绝
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}
