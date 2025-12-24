import React, { useState } from 'react';
import { Wrench, ChevronDown, ChevronUp, Clock, CheckCircle, XCircle } from 'lucide-react';
import { AgentToolCall } from '../services/api';

interface ToolExecutionViewProps {
  toolCalls: AgentToolCall[];
}

const ToolExecutionView: React.FC<ToolExecutionViewProps> = ({ toolCalls }) => {
  const [expandedCalls, setExpandedCalls] = useState<Set<string>>(new Set());

  if (!toolCalls || toolCalls.length === 0) {
    return null;
  }

  const toggleExpand = (id: string) => {
    const newExpanded = new Set(expandedCalls);
    if (newExpanded.has(id)) {
      newExpanded.delete(id);
    } else {
      newExpanded.add(id);
    }
    setExpandedCalls(newExpanded);
  };

  const formatDuration = (ms?: number) => {
    if (!ms) return '';
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const formatValue = (value: unknown): string => {
    if (typeof value === 'string') {
      if (value.length > 200) {
        return value.substring(0, 200) + '...';
      }
      return value;
    }
    try {
      const str = JSON.stringify(value, null, 2);
      if (str.length > 500) {
        return str.substring(0, 500) + '...';
      }
      return str;
    } catch {
      return String(value);
    }
  };

  return (
    <div className="mt-3 pt-3 border-t border-gray-200">
      <div className="flex items-center gap-2 mb-2 text-sm text-gray-600">
        <Wrench className="w-4 h-4" />
        <span className="font-medium">工具调用 ({toolCalls.length})</span>
      </div>

      <div className="space-y-2">
        {toolCalls.map((call) => {
          const isExpanded = expandedCalls.has(call.id);
          const hasError = !!call.error;

          return (
            <div
              key={call.id}
              className={`rounded-lg border ${
                hasError ? 'border-red-200 bg-red-50' : 'border-gray-200 bg-white'
              }`}
            >
              {/* Header */}
              <button
                onClick={() => toggleExpand(call.id)}
                className="w-full flex items-center justify-between p-2 text-left hover:bg-gray-50 rounded-t-lg"
              >
                <div className="flex items-center gap-2">
                  {hasError ? (
                    <XCircle className="w-4 h-4 text-red-500" />
                  ) : (
                    <CheckCircle className="w-4 h-4 text-green-500" />
                  )}
                  <span className="font-medium text-sm text-gray-800">
                    {call.name}
                  </span>
                  {call.duration_ms && (
                    <span className="flex items-center gap-1 text-xs text-gray-500">
                      <Clock className="w-3 h-3" />
                      {formatDuration(call.duration_ms)}
                    </span>
                  )}
                </div>
                {isExpanded ? (
                  <ChevronUp className="w-4 h-4 text-gray-400" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-gray-400" />
                )}
              </button>

              {/* Details */}
              {isExpanded && (
                <div className="px-3 pb-3 space-y-2 text-sm">
                  {/* Arguments */}
                  <div>
                    <span className="text-xs font-medium text-gray-500">输入参数:</span>
                    <pre className="mt-1 p-2 bg-gray-100 rounded text-xs overflow-x-auto">
                      {formatValue(call.args)}
                    </pre>
                  </div>

                  {/* Result or Error */}
                  {call.result !== undefined && (
                    <div>
                      <span className="text-xs font-medium text-gray-500">执行结果:</span>
                      <pre className="mt-1 p-2 bg-green-50 rounded text-xs overflow-x-auto text-green-800">
                        {formatValue(call.result)}
                      </pre>
                    </div>
                  )}

                  {call.error && (
                    <div>
                      <span className="text-xs font-medium text-red-500">错误:</span>
                      <pre className="mt-1 p-2 bg-red-100 rounded text-xs overflow-x-auto text-red-800">
                        {call.error}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ToolExecutionView;
