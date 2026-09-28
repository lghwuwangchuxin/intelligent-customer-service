/**
 * 推理模型输出处理。
 *
 * 🔴 qwen3 / deepseek-r1 这类推理模型会把思维链**混在 content 里**以
 *    `<think>…</think>` 包裹。直接渲染会让用户看到一大段自言自语，
 *    而且流式时标签是分片到达的，不能简单 replace。
 */
const OPEN = '<think>'
const CLOSE = '</think>'

export interface SplitContent {
  /** 思维链正文（可能尚未闭合） */
  thinking: string
  /** 真正要展示给用户的答案 */
  answer: string
  /** 思维链是否还在生成中 */
  thinkingOpen: boolean
}

export function splitThinking(raw: string): SplitContent {
  let thinking = ''
  let answer = ''
  let rest = raw

  while (true) {
    const start = rest.indexOf(OPEN)
    if (start === -1) {
      answer += rest
      break
    }
    answer += rest.slice(0, start)
    const after = rest.slice(start + OPEN.length)
    const end = after.indexOf(CLOSE)
    if (end === -1) {
      // 流式中标签还没闭合：剩下的全算思维链
      return { thinking: thinking + after, answer: answer.trim(), thinkingOpen: true }
    }
    thinking += after.slice(0, end)
    rest = after.slice(end + CLOSE.length)
  }

  return { thinking: thinking.trim(), answer: answer.trim(), thinkingOpen: false }
}

/**
 * 剥离模型自己吐出来的 `<retrieved_documents>` 块。
 *
 * 🔴 浏览器实跑抓到的真实问题：模型**不调用检索工具**，而是自己编造一段
 *    `<retrieved_documents>…</retrieved_documents>` 再据此作答。编出来的内容
 *    与知识库完全不符，但因为带着标签，看起来像是有出处的引用。
 *
 * 这个标签只会**由系统注入给模型**，永远不是合法的模型输出。
 * 服务端已在提示词里显式禁止，这里再做一道显示层兜底：
 * 宁可少显示一段，也不能把伪造的"引用"渲染成真的。
 */
const DOC_BLOCK = /<retrieved_documents>[\s\S]*?(?:<\/retrieved_documents>|$)/g

export interface StrippedContent {
  text: string
  /** 模型确实伪造了引用块——UI 应当提示这次回答缺乏可信出处 */
  fabricatedCitation: boolean
}

export function stripFabricatedDocs(raw: string): StrippedContent {
  if (!raw.includes('<retrieved_documents>')) {
    return { text: raw, fabricatedCitation: false }
  }
  return { text: raw.replace(DOC_BLOCK, '').trim(), fabricatedCitation: true }
}
