import { useCallback, useMemo, useState } from 'react'
import { isValidTenant, loadIdentity, saveIdentity, type Identity } from '../lib/tenant'

/**
 * 租户身份。
 *
 * 🔴 后端 fail-closed：没有租户就不该发请求。所以这里把「是否就绪」暴露出来，
 *    由 App 决定先渲染设置界面，而不是发一堆注定 400 的请求。
 */
export function useIdentity() {
  const [identity, setIdentity] = useState<Identity>(() => loadIdentity())

  const update = useCallback((next: Identity) => {
    saveIdentity(next)
    setIdentity(next)
  }, [])

  const ready = useMemo(() => isValidTenant(identity.tenantId), [identity.tenantId])

  return { identity, update, ready }
}
