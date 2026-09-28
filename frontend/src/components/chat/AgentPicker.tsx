import { ShieldAlert } from 'lucide-react'
import type { AgentInfo } from '../../api/types'
import { Badge, cx } from '../ui/Primitives'

/**
 * Agent 选择。留空 = 由服务端按问题内容自动分派。
 * 单 Agent 与多 Agent 在后端是同一套机制，前端也就只需要这一个控件。
 */
export function AgentPicker({
  agents,
  value,
  onChange,
}: {
  agents: AgentInfo[]
  value: string | null
  onChange: (id: string | null) => void
}) {
  return (
    <div className="space-y-1">
      <button
        onClick={() => onChange(null)}
        className={cx(
          'w-full rounded-lg border px-3 py-2 text-left text-sm transition',
          value === null
            ? 'border-primary-300 bg-primary-50 text-primary-800'
            : 'border-slate-200 bg-white hover:bg-slate-50',
        )}
      >
        <div className="font-medium">自动分派</div>
        <div className="mt-0.5 text-xs text-slate-500">由服务端按问题内容选择合适的智能体</div>
      </button>

      {agents.map((a) => (
        <button
          key={a.id}
          onClick={() => onChange(a.id)}
          className={cx(
            'w-full rounded-lg border px-3 py-2 text-left text-sm transition',
            value === a.id
              ? 'border-primary-300 bg-primary-50 text-primary-800'
              : 'border-slate-200 bg-white hover:bg-slate-50',
          )}
        >
          <div className="flex items-center gap-1.5 font-medium">
            {a.display_name || a.id}
            {a.requires_approval.length > 0 && (
              <ShieldAlert size={13} className="text-amber-500" />
            )}
          </div>
          <div className="mt-0.5 line-clamp-2 text-xs text-slate-500">{a.description}</div>
          <div className="mt-1.5 flex flex-wrap gap-1">
            {a.tools.map((t) => (
              <Badge key={t}>{t}</Badge>
            ))}
          </div>
        </button>
      ))}
    </div>
  )
}
