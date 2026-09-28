/**
 * 图节点名 → 用户可读标签。
 *
 * 服务端流出的是编排框架的真实节点名（如 `PIIMiddleware[cn_mobile].before_model`）。
 * 这些名字对排障很有价值，但直接显示给客服人员没有意义。
 */
const RULES: [RegExp, string][] = [
  [/^PIIMiddleware\[(.+?)\]/, '正在脱敏'],
  [/^ModelCallLimitMiddleware/, '检查调用上限'],
  [/^ToolCallLimitMiddleware/, '检查工具调用上限'],
  [/^SummarizationMiddleware/, '整理上下文'],
  [/^ToolRetryMiddleware/, '重试工具'],
  [/^HumanInTheLoopMiddleware/, '等待人工确认'],
  [/^tools$/, '调用工具'],
  [/^model$/, '正在思考'],
]

export function nodeLabel(node: string): string {
  for (const [re, label] of RULES) {
    if (re.test(node)) return label
  }
  return '处理中'
}
