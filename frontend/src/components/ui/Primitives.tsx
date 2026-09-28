import type { ReactNode } from 'react'
import { AlertTriangle, Loader2, RefreshCw } from 'lucide-react'
import type { ApiError } from '../../lib/errors'

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ')
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-500">
      <Loader2 size={16} className="animate-spin" />
      {label ?? '加载中…'}
    </div>
  )
}

/**
 * 错误展示。
 * 🔴 直接显示后端的 message（已本地化），只在 retryable 时才给重试按钮——
 *    对不可重试的错误（参数错、越权、预算超）给重试只会浪费一次调用。
 */
export function ErrorBox({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm">
      <div className="flex items-start gap-2">
        <AlertTriangle size={16} className="mt-0.5 shrink-0 text-red-500" />
        <div className="min-w-0 flex-1">
          <p className="font-medium text-red-800">{error.message}</p>
          {error.detail && <p className="mt-1 break-words text-red-600">{error.detail}</p>}
          <p className="mt-1 font-mono text-xs text-red-400">{error.code}</p>
        </div>
        {error.retryable && onRetry && (
          <button
            onClick={onRetry}
            className="flex shrink-0 items-center gap-1 rounded-md border border-red-300 px-2 py-1 text-xs text-red-700 hover:bg-red-100"
          >
            <RefreshCw size={12} /> 重试
          </button>
        )}
      </div>
    </div>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="py-10 text-center text-sm text-slate-400">{children}</div>
}

export function Badge({
  children,
  tone = 'slate',
}: {
  children: ReactNode
  tone?: 'slate' | 'green' | 'amber' | 'red' | 'blue'
}) {
  const tones = {
    slate: 'bg-slate-100 text-slate-600',
    green: 'bg-green-100 text-green-700',
    amber: 'bg-amber-100 text-amber-700',
    red: 'bg-red-100 text-red-700',
    blue: 'bg-primary-100 text-primary-700',
  }
  return (
    <span className={cx('rounded px-1.5 py-0.5 text-xs font-medium', tones[tone])}>
      {children}
    </span>
  )
}

export function Modal({
  title,
  onClose,
  children,
  wide,
}: {
  title: string
  onClose: () => void
  children: ReactNode
  wide?: boolean
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={onClose}
    >
      <div
        className={cx(
          'flex max-h-[85vh] w-full flex-col overflow-hidden rounded-xl bg-white shadow-xl',
          wide ? 'max-w-4xl' : 'max-w-2xl',
        )}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between border-b border-slate-200 px-5 py-3">
          <h2 className="font-semibold text-slate-800">{title}</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
            ✕
          </button>
        </header>
        <div className="flex-1 overflow-y-auto p-5">{children}</div>
      </div>
    </div>
  )
}
