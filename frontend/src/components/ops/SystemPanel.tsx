import { opsApi } from '../../api/ops'
import { useAsync } from '../../hooks/useAsync'
import { Badge, ErrorBox, Spinner } from '../ui/Primitives'

/**
 * 系统信息。取代旧版的「模型配置」与「服务状态」两块：
 * 模型现在由环境变量的别名表决定，前端只读不改；
 * 服务也从 9 个收敛成 1 个 API 进程，没有状态面板可看了。
 */
export function SystemPanel() {
  const info = useAsync(() => opsApi.systemInfo(), [])
  const ready = useAsync(() => opsApi.ready(), [])

  if (info.loading) return <Spinner />
  if (info.error) return <ErrorBox error={info.error} onRetry={() => void info.reload()} />
  if (!info.data) return null

  const d = info.data

  return (
    <div className="space-y-5 text-sm">
      <Section title="运行状态">
        <Row label="环境">
          <Badge tone={d.env === 'prod' ? 'green' : 'slate'}>{d.env}</Badge>
        </Row>
        <Row label="数据库">
          {ready.data?.database === 'ok' ? (
            <Badge tone="green">正常</Badge>
          ) : (
            <Badge tone="red">{ready.data?.database ?? '未知'}</Badge>
          )}
        </Row>
        <Row label="会话记忆">
          {d.durable_memory ? (
            <Badge tone="green">持久化</Badge>
          ) : (
            /* 🔴 降级为进程内：重启即丢、多副本不共享，线上表现是"用户偶尔失忆" */
            <Badge tone="amber">进程内（重启丢失）</Badge>
          )}
        </Row>
      </Section>

      <Section title="模型别名">
        <p className="mb-2 text-xs text-slate-500">
          业务代码只认别名，换厂商改环境变量即可，不需要改代码。
        </p>
        {Object.entries(d.models).map(([alias, model]) => (
          <Row key={alias} label={alias}>
            <span className="font-mono text-xs text-slate-600">{model}</span>
          </Row>
        ))}
      </Section>

      <Section title="资产">
        <Row label="智能体">
          <div className="flex flex-wrap justify-end gap-1">
            {d.agents.map((a) => (
              <Badge key={a}>{a}</Badge>
            ))}
          </div>
        </Row>
        <Row label="提示词">
          <div className="flex flex-wrap justify-end gap-1">
            {d.prompts.map((p) => (
              <Badge key={p}>{p}</Badge>
            ))}
          </div>
        </Row>
        <Row label="工具">
          <div className="flex flex-wrap justify-end gap-1">
            {d.tools.map((t) => (
              <Badge key={t}>{t}</Badge>
            ))}
          </div>
        </Row>
      </Section>

      <Section title="检索参数">
        {Object.entries(d.retrieval).map(([k, v]) => (
          <Row key={k} label={k}>
            <span className="font-mono text-xs text-slate-600">{String(v)}</span>
          </Row>
        ))}
      </Section>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h3 className="mb-2 text-sm font-semibold text-slate-700">{title}</h3>
      <div className="divide-y divide-slate-100 rounded-lg border border-slate-200 bg-white">
        {children}
      </div>
    </section>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 px-3 py-2">
      <span className="shrink-0 text-slate-600">{label}</span>
      <div className="min-w-0 text-right">{children}</div>
    </div>
  )
}
