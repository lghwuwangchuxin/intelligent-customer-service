import { describe, expect, it } from 'vitest'
import { splitThinking, stripFabricatedDocs } from '../reasoning'

describe('splitThinking', () => {
  it('普通内容原样返回', () => {
    expect(splitThinking('你好')).toEqual({ thinking: '', answer: '你好', thinkingOpen: false })
  })

  it('拆出闭合的思维链', () => {
    const r = splitThinking('<think>盘算一下</think>答案是 2')
    expect(r.thinking).toBe('盘算一下')
    expect(r.answer).toBe('答案是 2')
    expect(r.thinkingOpen).toBe(false)
  })

  it('🔴 流式中标签尚未闭合时，剩余内容全算思维链', () => {
    // 这是最容易出错的情形：token 逐个到达，</think> 还没来
    const r = splitThinking('<think>正在盘算')
    expect(r.thinking).toBe('正在盘算')
    expect(r.answer).toBe('')
    expect(r.thinkingOpen).toBe(true)
  })

  it('思维链在答案之后也能拆', () => {
    const r = splitThinking('先说结论<think>再解释</think>')
    expect(r.answer).toBe('先说结论')
    expect(r.thinking).toBe('再解释')
  })

  it('多段思维链合并', () => {
    const r = splitThinking('<think>一</think>A<think>二</think>B')
    expect(r.thinking).toBe('一二')
    expect(r.answer).toBe('AB')
  })

  it('逐 token 累积时不会把答案吞掉', () => {
    // 模拟流式：每次都对累积的全量文本重新拆分
    const chunks = ['<think>', '想', '一下', '</think>', '答案', '是 2']
    let acc = ''
    const seen: string[] = []
    for (const c of chunks) {
      acc += c
      seen.push(splitThinking(acc).answer)
    }
    expect(seen.at(-1)).toBe('答案是 2')
    // 思维链未闭合期间，答案必须一直是空的，不能漏出思维内容
    expect(seen.slice(0, 3)).toEqual(['', '', ''])
  })
})

describe('stripFabricatedDocs', () => {
  it('普通回答不受影响', () => {
    const r = stripFabricatedDocs('退款 3 个工作日到账')
    expect(r.text).toBe('退款 3 个工作日到账')
    expect(r.fabricatedCitation).toBe(false)
  })

  it('🔴 剥掉模型伪造的引用块并示警', () => {
    // 浏览器实跑抓到的真实输出形态：模型没调工具，自己编了一段带标签的"资料"
    const raw = '<retrieved_documents>[{"content":"编造的内容"}]</retrieved_documents>\n\n真正的回答'
    const r = stripFabricatedDocs(raw)
    expect(r.text).toBe('真正的回答')
    expect(r.fabricatedCitation).toBe(true)
  })

  it('未闭合的伪造块也要剥掉（流式中途）', () => {
    const r = stripFabricatedDocs('<retrieved_documents>[{"content":"半截')
    expect(r.text).toBe('')
    expect(r.fabricatedCitation).toBe(true)
  })

  it('多个伪造块全部剥掉', () => {
    const r = stripFabricatedDocs('<retrieved_documents>a</retrieved_documents>中间<retrieved_documents>b</retrieved_documents>尾')
    expect(r.text).toBe('中间\n\n尾'.replace('\n\n', ''))
    expect(r.fabricatedCitation).toBe(true)
  })
})
