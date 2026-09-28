import { request } from '../lib/http'
import type { MemoryItem, PendingAction, Readiness, SystemInfo } from './types'

export const opsApi = {
  /** 待人工确认的写操作。后端把写操作转成操作单，没有绕过路径。 */
  pending: () => request<{ pending: PendingAction[] }>('/pending').then((r) => r.pending),

  resolve: (id: string, status: 'approved' | 'rejected') =>
    request<{ updated: boolean }>(`/pending/${id}`, { method: 'POST', body: { status } }),

  memories: (q?: string) =>
    request<{ memories: MemoryItem[] }>('/memories', { query: { q } }).then((r) => r.memories),

  systemInfo: () => request<SystemInfo>('/system/info'),

  /** readiness 才查依赖；/health 是 liveness，不查（查了会把故障放大成雪崩）。 */
  ready: () => request<Readiness>('/ready'),
}
