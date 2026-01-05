import React from 'react';
import ReactMarkdown from 'react-markdown';
import { User, Bot, Brain, Users, Car, Battery, Wallet, AlertTriangle, BarChart3, Wrench, Leaf, Calendar } from 'lucide-react';
import ToolExecutionView from './ToolExecutionView';
import { AgentToolCall, AgentThought, A2ARoutingMode } from '../services/api';

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
  // Multi-agent fields
  isMultiAgentResponse?: boolean;
  agentName?: string;
  agentDisplayName?: string;
  routingMode?: A2ARoutingMode;
  parallelResults?: Array<{
    agent: string;
    agentDisplayName: string;
    response: string;
  }>;
  chainSteps?: Array<{
    agent: string;
    agentDisplayName: string;
    response: string;
    step: number;
  }>;
}

// Agent icons mapping
const AGENT_ICONS: Record<string, React.ElementType> = {
  travel_assistant: Car,
  charging_manager: Battery,
  billing_advisor: Wallet,
  emergency_support: AlertTriangle,
  data_analyst: BarChart3,
  maintenance_expert: Wrench,
  energy_advisor: Leaf,
  scheduling_advisor: Calendar,
};

const AGENT_COLORS: Record<string, string> = {
  travel_assistant: 'bg-blue-500',
  charging_manager: 'bg-green-500',
  billing_advisor: 'bg-yellow-500',
  emergency_support: 'bg-red-500',
  data_analyst: 'bg-purple-500',
  maintenance_expert: 'bg-orange-500',
  energy_advisor: 'bg-emerald-500',
  scheduling_advisor: 'bg-indigo-500',
};

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const isAgent = message.isAgentResponse;
  const isMultiAgent = message.isMultiAgentResponse;

  // Get the appropriate icon for multi-agent responses
  const getAgentIcon = () => {
    if (isMultiAgent && message.agentName) {
      const IconComponent = AGENT_ICONS[message.agentName];
      if (IconComponent) {
        return <IconComponent className="w-5 h-5 text-white" />;
      }
    }
    if (isMultiAgent) {
      return <Users className="w-5 h-5 text-white" />;
    }
    if (isAgent) {
      return <Brain className="w-5 h-5 text-white" />;
    }
    return <Bot className="w-5 h-5 text-white" />;
  };

  // Get the appropriate background color
  const getAvatarColor = () => {
    if (isUser) return 'bg-primary-500';
    if (isMultiAgent && message.agentName) {
      return AGENT_COLORS[message.agentName] || 'bg-indigo-500';
    }
    if (isMultiAgent) return 'bg-indigo-500';
    if (isAgent) return 'bg-purple-500';
    return 'bg-green-500';
  };

  // Get the display name
  const getDisplayName = () => {
    if (isUser) return '您';
    if (isMultiAgent && message.agentDisplayName) {
      return message.agentDisplayName;
    }
    if (isMultiAgent) return '多智能体';
    if (isAgent) return 'Agent';
    return '智能客服';
  };

  // Get background color for message container
  const getContainerBg = () => {
    if (isUser) return 'bg-white';
    if (isMultiAgent) return 'bg-indigo-50';
    if (isAgent) return 'bg-purple-50';
    return 'bg-gray-50';
  };

  return (
    <div className={`flex gap-3 p-4 message-fade-in ${getContainerBg()}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${getAvatarColor()}`}
      >
        {isUser ? <User className="w-5 h-5 text-white" /> : getAgentIcon()}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-gray-900">{getDisplayName()}</span>
          <span className="text-xs text-gray-400">
            {message.timestamp.toLocaleTimeString()}
          </span>
          {isAgent && message.iterations && (
            <span className="text-xs bg-purple-100 text-purple-700 px-1.5 py-0.5 rounded">
              {message.iterations} 轮推理
            </span>
          )}
          {isMultiAgent && message.routingMode && (
            <span className="text-xs bg-indigo-100 text-indigo-700 px-1.5 py-0.5 rounded">
              {message.routingMode === 'auto'
                ? '自动路由'
                : message.routingMode === 'parallel'
                ? '并行调用'
                : '串行调用'}
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
