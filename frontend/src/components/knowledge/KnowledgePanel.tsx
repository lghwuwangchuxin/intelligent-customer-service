import { useCallback, useRef, useState } from 'react'
import { FileText, Search, Trash2, Upload } from 'lucide-react'
import { knowledgeApi } from '../../api/knowledge'
import type { RetrieveResult } from '../../api/types'
import { ApiError } from '../../lib/errors'
import { useAsync } from '../../hooks/useAsync'
import { Badge, Empty, ErrorBox, Spinner, cx } from '../ui/Primitives'

export function KnowledgePanel() {
  const docs = useAsync(() => knowledgeApi.documents(), [])
  const stats = useAsync(() => knowledgeApi.stats(), [])

  const [query, setQuery] = useState('')
  const [result, setResult] = useState<RetrieveResult | null>(null)
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const reloadAll = useCallback(() => {
    void docs.reload()
    void stats.reload()
  }, [docs, stats])

  const search = useCallback(async () => {
    const q = query.trim()
    if (!q) return
    setSearching(true)
    setError(null)
    try {
      setResult(await knowledgeApi.retrieve(q))
    } catch (err) {
      if (err instanceof ApiError) setError(err)
      else throw err
    } finally {
      setSearching(false)
    }
  }, [query])

  const doUpload = useCallback(
    async (file: File) => {
      setError(null)
      try {
        await knowledgeApi.upload(file)
        reloadAll()
      } catch (err) {
        // 🔴 服务端只接受能真正解析的格式：传 PDF/Word 会走解析器，
        //    认不出的格式返回 UNSUPPORTED_DOCUMENT 而不是把二进制当文本入库
        if (err instanceof ApiError) setError(err)
        else throw err
      }
    },
    [reloadAll],
  )

  return (
    <div className="space-y-5">
      {/* 统计 */}
      {stats.data && (
        <div className="grid grid-cols-3 gap-3">
          <Stat label="文档" value={stats.data.documents} />
          <Stat label="切块" value={stats.data.chunks} />
          <RecallStat recall={stats.data.last_recall} />
        </div>
      )}

      {/* 检索 */}
      <section>
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && void search()}
            placeholder="检索知识库…"
            className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-primary-500 focus:outline-none"
          />
          <button
            onClick={() => void search()}
            disabled={searching || !query.trim()}
            className="flex items-center gap-1.5 rounded-lg bg-primary-600 px-4 text-sm text-white hover:bg-primary-700 disabled:bg-slate-300"
          >
            <Search size={15} /> 检索
          </button>
        </div>

        {searching && <div className="mt-3"><Spinner label="检索中…" /></div>}
        {error && <div className="mt-3"><ErrorBox error={error} onRetry={() => void search()} /></div>}

        {result && !searching && (
          <div className="mt-3 space-y-2">
            <div className="flex items-center gap-2 text-xs text-slate-500">
              {/* strategy 能看出词法路是否降级：hybrid → dense 说明 BM25 那一路挂了 */}
              <Badge tone={result.strategy.startsWith('hybrid') ? 'green' : 'amber'}>
                {result.strategy}
              </Badge>
              <span>{result.latency_ms} ms</span>
              <span>{result.documents.length} 条</span>
            </div>
            {result.documents.length === 0 && <Empty>没有检索到相关资料</Empty>}
            {result.documents.map((d) => (
              <div key={d.id} className="rounded-lg border border-slate-200 bg-white p-3">
                <div className="mb-1 flex items-center gap-2 text-xs text-slate-500">
                  <span className="font-mono">{d.doc_id}</span>
                  <span>#{d.chunk_no}</span>
                  {d.section && <span className="truncate">{d.section}</span>}
                  {d.score !== null && (
                    <span className="ml-auto shrink-0">{d.score.toFixed(3)}</span>
                  )}
                </div>
                <p className="line-clamp-4 text-sm text-slate-700">{d.content}</p>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* 文档列表 */}
      <section>
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-700">文档</h3>
          <button
            onClick={() => fileRef.current?.click()}
            className="flex items-center gap-1.5 rounded-lg border border-slate-300 px-2.5 py-1.5 text-xs text-slate-700 hover:bg-slate-50"
          >
            <Upload size={13} /> 上传
          </button>
          <input
            ref={fileRef}
            type="file"
            hidden
            accept=".pdf,.docx,.txt,.md,.html,.htm,.csv,.json"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) void doUpload(f)
              e.target.value = ''
            }}
          />
        </div>

        {docs.loading && <Spinner />}
        {docs.error && <ErrorBox error={docs.error} onRetry={() => void docs.reload()} />}
        {docs.data?.length === 0 && <Empty>知识库还是空的</Empty>}

        <div className="space-y-1">
          {docs.data?.map((d) => (
            <div
              key={d.doc_id}
              className="group flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2"
            >
              <FileText size={14} className="shrink-0 text-slate-400" />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm text-slate-700">{d.title || d.doc_id}</p>
                <p className="text-xs text-slate-400">
                  {d.chunk_count} 块 · {new Date(d.updated_at).toLocaleString('zh-CN')}
                </p>
              </div>
              <button
                onClick={async () => {
                  await knowledgeApi.remove(d.doc_id)
                  reloadAll()
                }}
                className="shrink-0 text-slate-300 opacity-0 transition hover:text-red-500 group-hover:opacity-100"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-0.5 text-xl font-semibold text-slate-800">{value}</p>
    </div>
  )
}

/**
 * 召回率基线。
 * 🔴 HNSW 索引退化完全无声——不报错、不变慢，只是开始漏结果。
 *    这个数字是唯一能看出来的地方，所以低于 0.9 要显著标红。
 */
function RecallStat({ recall }: { recall: { recall: number; at: string } | null }) {
  if (!recall) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-3">
        <p className="text-xs text-slate-500">召回率基线</p>
        <p className="mt-0.5 text-sm text-slate-400">尚未巡检</p>
      </div>
    )
  }
  const low = recall.recall < 0.9
  return (
    <div
      className={cx(
        'rounded-lg border p-3',
        low ? 'border-red-200 bg-red-50' : 'border-slate-200 bg-white',
      )}
    >
      <p className="text-xs text-slate-500">召回率基线</p>
      <p className={cx('mt-0.5 text-xl font-semibold', low ? 'text-red-600' : 'text-slate-800')}>
        {(recall.recall * 100).toFixed(1)}%
      </p>
      {low && <p className="text-xs text-red-500">偏低，建议调大 ef_search</p>}
    </div>
  )
}
