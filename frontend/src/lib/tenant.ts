/**
 * 租户。
 *
 * 🔴 后端是 fail-closed 的：任何业务请求缺少 X-Tenant-Id 一律 400。
 *    所以前端必须在发请求前就确保租户存在，而不是发出去再处理错误。
 */
const KEY = 'ics.tenant'
const USER_KEY = 'ics.user'

export interface Identity {
  tenantId: string
  userId: string
}

function read(key: string): string {
  try {
    return localStorage.getItem(key) ?? ''
  } catch {
    // 隐私模式 / 禁用存储时 localStorage 会抛，不能让它把应用打挂
    return ''
  }
}

function write(key: string, value: string): void {
  try {
    localStorage.setItem(key, value)
  } catch {
    /* 忽略：存不下只是记不住，不影响本次会话 */
  }
}

export function loadIdentity(): Identity {
  return { tenantId: read(KEY), userId: read(USER_KEY) }
}

export function saveIdentity(identity: Identity): void {
  write(KEY, identity.tenantId)
  write(USER_KEY, identity.userId)
}

/** 与后端 core/context.py 的校验保持一致，避免把必然 400 的请求发出去。 */
const TENANT_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/

export function isValidTenant(value: string): boolean {
  return TENANT_RE.test(value)
}
