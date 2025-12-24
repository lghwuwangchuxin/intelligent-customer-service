import React, { useState, useEffect } from 'react';
import { Brain, Wrench, Settings, ChevronDown, ChevronUp } from 'lucide-react';
import { mcpApi, MCPTool } from '../services/api';

interface AgentPanelProps {
  isAgentMode: boolean;
  onToggleAgentMode: (enabled: boolean) => void;
  currentStep?: number;
  status?: string;
}

const AgentPanel: React.FC<AgentPanelProps> = ({
  isAgentMode,
  onToggleAgentMode,
  currentStep,
  status,
}) => {
  const [showTools, setShowTools] = useState(false);
  const [tools, setTools] = useState<MCPTool[]>([]);
  const [loadingTools, setLoadingTools] = useState(false);

  useEffect(() => {
    if (showTools && tools.length === 0) {
      loadTools();
    }
  }, [showTools]);

  const loadTools = async () => {
    setLoadingTools(true);
    try {
      const toolList = await mcpApi.listTools();
      setTools(toolList);
    } catch (err) {
      console.error('Failed to load tools:', err);
    } finally {
      setLoadingTools(false);
    }
  };

  return (
    <div className="bg-white border-b border-gray-200 px-4 py-2">
      <div className="max-w-4xl mx-auto flex items-center justify-between">
        {/* Agent Mode Toggle */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => onToggleAgentMode(!isAgentMode)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
              isAgentMode
                ? 'bg-purple-100 text-purple-700 hover:bg-purple-200'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            <Brain className="w-4 h-4" />
            {isAgentMode ? 'Agent 模式' : '普通模式'}
          </button>

          {isAgentMode && (
            <div className="flex items-center gap-2 text-sm text-gray-500">
              {status && (
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                  {status}
                </span>
              )}
              {currentStep !== undefined && currentStep > 0 && (
                <span className="text-xs bg-gray-100 px-2 py-0.5 rounded">
                  Step {currentStep}
                </span>
              )}
            </div>
          )}
        </div>

        {/* Tools Info */}
        {isAgentMode && (
          <button
            onClick={() => setShowTools(!showTools)}
            className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700"
          >
            <Wrench className="w-4 h-4" />
            <span>工具</span>
            {showTools ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>
        )}
      </div>

      {/* Tools List */}
      {showTools && isAgentMode && (
        <div className="max-w-4xl mx-auto mt-2 p-3 bg-gray-50 rounded-lg">
          <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
            <Settings className="w-4 h-4" />
            可用工具
          </h4>
          {loadingTools ? (
            <p className="text-sm text-gray-500">加载中...</p>
          ) : tools.length === 0 ? (
            <p className="text-sm text-gray-500">暂无可用工具</p>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              {tools.map((tool) => (
                <div
                  key={tool.name}
                  className="bg-white p-2 rounded border border-gray-200 text-sm"
                >
                  <div className="font-medium text-gray-800">{tool.name}</div>
                  <div className="text-xs text-gray-500 line-clamp-2">
                    {tool.description}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AgentPanel;
