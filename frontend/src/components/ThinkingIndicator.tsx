import React from 'react';
import { Brain, Wrench, Eye, MessageSquare, Loader2 } from 'lucide-react';

export interface ThinkingStep {
  type: 'thinking' | 'action' | 'observation' | 'response';
  content: string;
  tool?: string;
  step?: number;
}

interface ThinkingIndicatorProps {
  steps: ThinkingStep[];
  currentStatus?: string;
  isLoading?: boolean;
}

const ThinkingIndicator: React.FC<ThinkingIndicatorProps> = ({
  steps,
  currentStatus,
  isLoading,
}) => {
  const getStepIcon = (type: ThinkingStep['type']) => {
    switch (type) {
      case 'thinking':
        return <Brain className="w-4 h-4 text-purple-500" />;
      case 'action':
        return <Wrench className="w-4 h-4 text-blue-500" />;
      case 'observation':
        return <Eye className="w-4 h-4 text-green-500" />;
      case 'response':
        return <MessageSquare className="w-4 h-4 text-gray-500" />;
      default:
        return <Loader2 className="w-4 h-4 animate-spin" />;
    }
  };

  const getStepLabel = (type: ThinkingStep['type']) => {
    switch (type) {
      case 'thinking':
        return '思考';
      case 'action':
        return '执行';
      case 'observation':
        return '观察';
      case 'response':
        return '回复';
      default:
        return '处理中';
    }
  };

  const getStepBgColor = (type: ThinkingStep['type']) => {
    switch (type) {
      case 'thinking':
        return 'bg-purple-50 border-purple-200';
      case 'action':
        return 'bg-blue-50 border-blue-200';
      case 'observation':
        return 'bg-green-50 border-green-200';
      case 'response':
        return 'bg-gray-50 border-gray-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  if (steps.length === 0 && !isLoading) {
    return null;
  }

  return (
    <div className="p-4 bg-gray-50 border-t border-gray-200">
      {/* Current status */}
      {isLoading && currentStatus && (
        <div className="flex items-center gap-2 mb-3 text-sm text-gray-600">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span>{currentStatus}</span>
        </div>
      )}

      {/* Steps timeline */}
      {steps.length > 0 && (
        <div className="space-y-2">
          {steps.map((step, index) => (
            <div
              key={index}
              className={`flex gap-3 p-2 rounded-lg border ${getStepBgColor(step.type)}`}
            >
              <div className="flex-shrink-0 mt-0.5">
                {getStepIcon(step.type)}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-medium text-gray-700">
                    {step.step && `#${step.step} `}
                    {getStepLabel(step.type)}
                  </span>
                  {step.tool && (
                    <span className="text-xs bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded">
                      {step.tool}
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-600 break-words">
                  {step.content}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Simple loading indicator when no steps yet */}
      {isLoading && steps.length === 0 && (
        <div className="flex items-center gap-2">
          <div className="flex gap-1">
            <span className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
          <span className="text-sm text-gray-500">Agent 思考中...</span>
        </div>
      )}
    </div>
  );
};

export default ThinkingIndicator;
