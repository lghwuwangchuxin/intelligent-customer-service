import React, { useState } from 'react';
import { Wrench, CheckCircle, XCircle, ChevronDown, ChevronRight, Clock } from 'lucide-react';
import { AgentToolCall } from '../services/api';

interface ToolExecutionViewProps {
  toolCalls: AgentToolCall[];
}

const ToolExecutionView: React.FC<ToolExecutionViewProps> = ({ toolCalls }) => {
  const [expandedCalls, setExpandedCalls] = useState<Set<string>>(new Set());

  const toggleExpand = (id: string) => {
    setExpandedCalls((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  if (!toolCalls || toolCalls.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 pt-3 border-t border-gray-200">
      <div className="text-xs text-gray-500 mb-2 flex items-center gap-1">
        <Wrench size={12} />
        工具调用 ({toolCalls.length})
      </div>
      <div className="space-y-2">
        {toolCalls.map((call, index) => {
          const callId = call.id || `tool_${index}`;
          const isExpanded = expandedCalls.has(callId);
          const isSuccess = call.status === 'success' || (!call.error && call.result !== undefined);
          const isError = call.status === 'error' || !!call.error;

          return (
            <div
              key={callId}
              className={`rounded-lg border ${
                isError ? 'border-red-200 bg-red-50' : 'border-gray-200 bg-white'
              }`}
            >
              <button
                onClick={() => toggleExpand(callId)}
                className="w-full px-3 py-2 flex items-center justify-between text-left"
              >
                <div className="flex items-center gap-2">
                  {isSuccess ? (
                    <CheckCircle size={14} className="text-green-500" />
                  ) : isError ? (
                    <XCircle size={14} className="text-red-500" />
                  ) : (
                    <Clock size={14} className="text-yellow-500 animate-spin" />
                  )}
                  <span className="text-sm font-medium text-gray-900">{call.name}</span>
                  {call.duration_ms && (
                    <span className="text-xs text-gray-400">{call.duration_ms}ms</span>
                  )}
                </div>
                {isExpanded ? (
                  <ChevronDown size={14} className="text-gray-400" />
                ) : (
                  <ChevronRight size={14} className="text-gray-400" />
                )}
              </button>

              {isExpanded && (
                <div className="px-3 pb-3 space-y-2">
                  {/* Arguments */}
                  {call.args && Object.keys(call.args).length > 0 && (
                    <div>
                      <div className="text-xs text-gray-500 mb-1">参数:</div>
                      <pre className="text-xs bg-gray-100 p-2 rounded overflow-x-auto">
                        {JSON.stringify(call.args, null, 2)}
                      </pre>
                    </div>
                  )}

                  {/* Result */}
                  {call.result !== undefined && (
                    <div>
                      <div className="text-xs text-gray-500 mb-1">结果:</div>
                      <pre className="text-xs bg-gray-100 p-2 rounded overflow-x-auto max-h-40 overflow-y-auto">
                        {typeof call.result === 'string'
                          ? call.result
                          : JSON.stringify(call.result, null, 2)}
                      </pre>
                    </div>
                  )}

                  {/* Error */}
                  {call.error && (
                    <div>
                      <div className="text-xs text-red-500 mb-1">错误:</div>
                      <pre className="text-xs bg-red-100 text-red-700 p-2 rounded overflow-x-auto">
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
