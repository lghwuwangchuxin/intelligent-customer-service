import { useState } from 'react'
import Markdown from 'react-markdown'
import {
  AlertTriangle,
  Bot,
  Brain,
  ChevronDown,
  ChevronRight,
  ShieldAlert,
  User,
  Wrench,
} from 'lucide-react'
import type { Message } from '../../hooks/useChat'
import { splitThinking, stripFabricatedDocs } from '../../lib/reasoning'
import { Badge, ErrorBox, cx } from '../ui/Primitives'

export function ChatMessage({ message, agentName }: { message: Message; agentName?: string }) {
  const isUser = message.role === 'user'
  const [showThinking, setShowThinking] = useState(false)
  // 🔴 推理模型把思维链混在 content 里，必须拆开——否则用户看到一大段自言自语
  const { thinking, answer: rawAnswer, thinkingOpen } = isUser
    ? { thinking: '', answer: message.content, thinkingOpen: false }
    : splitThinking(message.content)
  // 🔴 模型可能伪造 <retrieved_documents> 块冒充出处，必须剔掉并示警
  const { text: answer, fabricatedCitation } = isUser
    ? { text: rawAnswer, fabricatedCitation: false }
    : stripFabricatedDocs(rawAnswer)

  return (
    <div className={cx('flex gap-3', isUser && 'flex-row-reverse')}>
      <div
        className={cx(
          'flex h-8 w-8 shrink-0 items-center justify-center rounded-full',
          isUser ? 'bg-primary-600 text-white' : 'bg-slate-200 text-slate-600',
        )}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>

      <div className={cx('min-w-0 max-w-[80%] space-y-2', isUser && 'items-end')}>
        {!isUser && agentName && (
          <div className="flex items-center gap-2">
            <Badge tone="blue">{agentName}</Badge>
          </div>
        )}

        <div
          className={cx(
            'rounded-xl px-4 py-2.5 text-sm',
            isUser ? 'bg-primary-600 text-white' : 'border border-slate-200 bg-white text-slate-800',
          )}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap break-words">{message.content}</p>
          ) : (
            <div className="markdown-content break-words">
              {answer ? (
                <Markdown>{answer}</Markdown>
              ) : (
                thinkingOpen && <span className="text-slate-400">正在思考…</span>
              )}
              {message.streaming && (
                <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-slate-400 align-middle" />
              )}
            </div>
          )}
        </div>

        {thinking && (
          <div className="rounded-lg border border-slate-200 bg-slate-50 text-xs">
            <button
              onClick={() => setShowThinking((v) => !v)}
              className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-slate-500 hover:text-slate-700"
            >
              {showThinking ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              <Brain size={12} />
              <span>思考过程{thinkingOpen ? '（进行中）' : ''}</span>
            </button>
            {showThinking && (
              <p className="whitespace-pre-wrap break-words border-t border-slate-200 px-2.5 py-2 text-slate-500">
                {thinking}
              </p>
            )}
          </div>
        )}

        {fabricatedCitation && (
          <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-2.5 text-sm text-red-800">
            <AlertTriangle size={16} className="mt-0.5 shrink-0" />
            <span>
              模型自行编造了资料引用（未真正检索知识库），该段已被移除。
              本次回答缺乏可信出处，请勿直接采信。
            </span>
          </div>
        )}

        {message.interrupted && (
          <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 p-2.5 text-sm text-amber-800">
            <ShieldAlert size={16} className="mt-0.5 shrink-0" />
            <span>
              该操作涉及写入，已转为待确认清单等待人工批准——服务端不会直接执行。
            </span>
          </div>
        )}

        {message.toolCalls && message.toolCalls.length > 0 && (
          <div className="space-y-1">
            {message.toolCalls.map((call, i) => (
              <div
                key={i}
                className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs text-slate-600"
              >
                <Wrench size={12} className="shrink-0 text-slate-400" />
                <span className="font-mono">{call.name}</span>
              </div>
            ))}
          </div>
        )}

        {message.error && <ErrorBox error={message.error} />}
      </div>
    </div>
  )
}
