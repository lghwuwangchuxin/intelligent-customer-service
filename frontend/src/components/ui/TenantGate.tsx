import { useState } from 'react'
import { Building2 } from 'lucide-react'
import { isValidTenant, type Identity } from '../../lib/tenant'

/**
 * 租户设置。
 *
 * 🔴 后端 fail-closed：缺 X-Tenant-Id 一律 400。与其让用户看到满屏报错，
 *    不如在进入应用前先把租户要到——这是前端对 fail-closed 的正确配合。
 */
export function TenantGate({
  identity,
  onSubmit,
}: {
  identity: Identity
  onSubmit: (next: Identity) => void
}) {
  const [tenantId, setTenantId] = useState(identity.tenantId || 'demo')
  const [userId, setUserId] = useState(identity.userId || 'u1')
  const valid = isValidTenant(tenantId)

  return (
    <div className="flex h-screen items-center justify-center bg-slate-50 p-4">
      <form
        onSubmit={(e) => {
          e.preventDefault()
          if (valid) onSubmit({ tenantId, userId })
        }}
        className="w-full max-w-md rounded-xl border border-slate-200 bg-white p-6 shadow-sm"
      >
        <div className="mb-4 flex items-center gap-2">
          <Building2 className="text-primary-600" size={20} />
          <h1 className="text-lg font-semibold text-slate-800">选择租户</h1>
        </div>
        <p className="mb-5 text-sm text-slate-500">
          所有接口都按租户隔离，服务端在缺少租户标识时会直接拒绝请求。
        </p>

        <label className="mb-1 block text-sm font-medium text-slate-700">租户标识</label>
        <input
          value={tenantId}
          onChange={(e) => setTenantId(e.target.value)}
          placeholder="demo"
          className="mb-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-primary-500 focus:outline-none"
        />
        {!valid && tenantId.length > 0 && (
          <p className="mb-2 text-xs text-red-500">
            只允许字母、数字、下划线与连字符，且不能以连字符开头，最长 64 位
          </p>
        )}

        <label className="mt-3 mb-1 block text-sm font-medium text-slate-700">
          用户标识<span className="ml-1 font-normal text-slate-400">（长期记忆需要）</span>
        </label>
        <input
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          placeholder="u1"
          className="mb-5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-primary-500 focus:outline-none"
        />

        <button
          type="submit"
          disabled={!valid}
          className="w-full rounded-lg bg-primary-600 py-2 text-sm font-medium text-white hover:bg-primary-700 disabled:bg-slate-300"
        >
          进入
        </button>
      </form>
    </div>
  )
}
