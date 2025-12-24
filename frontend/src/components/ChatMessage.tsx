import React from 'react';
import ReactMarkdown from 'react-markdown';
import { User, Bot, Brain } from 'lucide-react';
import ToolExecutionView from './ToolExecutionView';
import { AgentToolCall, AgentThought } from '../services/api';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: Array<{
    content: string;
    source: string;
  }>;
  // Agent mode fields
  isAgentResponse?: boolean;
  thoughts?: AgentThought[];
  toolCalls?: AgentToolCall[];
  iterations?: number;
}

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const isAgent = message.isAgentResponse;

  return (
    <div
      className={`flex gap-3 p-4 message-fade-in ${
        isUser ? 'bg-white' : isAgent ? 'bg-purple-50' : 'bg-gray-50'
      }`}
    >
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser ? 'bg-primary-500' : isAgent ? 'bg-purple-500' : 'bg-green-500'
        }`}
      >
        {isUser ? (
          <User className="w-5 h-5 text-white" />
        ) : isAgent ? (
          <Brain className="w-5 h-5 text-white" />
        ) : (
          <Bot className="w-5 h-5 text-white" />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-gray-900">
            {isUser ? '您' : isAgent ? 'Agent' : '智能客服'}
          </span>
          <span className="text-xs text-gray-400">
            {message.timestamp.toLocaleTimeString()}
          </span>
          {isAgent && message.iterations && (
            <span className="text-xs bg-purple-100 text-purple-700 px-1.5 py-0.5 rounded">
              {message.iterations} 轮推理
            </span>
          )}
        </div>

        <div className="prose prose-sm max-w-none text-gray-700">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>

        {/* Tool Calls (Agent mode) */}
        {message.toolCalls && message.toolCalls.length > 0 && (
          <ToolExecutionView toolCalls={message.toolCalls} />
        )}

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-200">
            <div className="text-xs text-gray-500 mb-2">参考来源:</div>
            <div className="space-y-2">
              {message.sources.map((source, index) => (
                <div
                  key={index}
                  className="text-xs bg-gray-100 rounded p-2 text-gray-600"
                >
                  <span className="font-medium">{source.source}</span>
                  <p className="mt-1 line-clamp-2">{source.content}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
