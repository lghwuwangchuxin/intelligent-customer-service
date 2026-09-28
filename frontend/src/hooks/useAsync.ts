import { useCallback, useEffect, useState } from 'react'
import { ApiError } from '../lib/errors'

interface State<T> {
  data: T | null
  error: ApiError | null
  loading: boolean
}

/** 加载远端数据的通用钩子。错误按后端契约归一，不各处重复 try/catch。 */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = [], enabled = true) {
  const [state, setState] = useState<State<T>>({ data: null, error: null, loading: enabled })

  const run = useCallback(async () => {
    if (!enabled) return
    setState((s) => ({ ...s, loading: true }))
    try {
      setState({ data: await fn(), error: null, loading: false })
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      setState({
        data: null,
        error: err instanceof ApiError ? err : null,
        loading: false,
      })
      if (!(err instanceof ApiError)) throw err
    }
    // fn 每次渲染都是新引用，依赖交给调用方用 deps 显式声明
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, ...deps])

  useEffect(() => {
    void run()
  }, [run])

  return { ...state, reload: run }
}
