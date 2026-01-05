import React, { useEffect, useState, useCallback } from 'react';
import {
  Users,
  Zap,
  ArrowRight,
  CheckCircle,
  XCircle,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Car,
  Battery,
  Wallet,
  AlertTriangle,
  BarChart3,
  Wrench,
  Leaf,
  Calendar,
} from 'lucide-react';
import { a2aApi, A2AAgentInfo, A2ARoutingMode } from '../services/api';

interface MultiAgentPanelProps {
  isMultiAgentMode: boolean;
  onToggleMultiAgentMode: (enabled: boolean) => void;
  routingMode: A2ARoutingMode;
  onRoutingModeChange: (mode: A2ARoutingMode) => void;
  selectedAgents: string[];
  onSelectedAgentsChange: (agents: string[]) => void;
}

// Agent metadata with icons and descriptions
const AGENT_METADATA: Record<string, { icon: React.ElementType; color: string; domain: string }> = {
  travel_assistant: { icon: Car, color: 'text-blue-500', domain: '充电桩运营' },
  charging_manager: { icon: Battery, color: 'text-green-500', domain: '充电桩运营' },
  billing_advisor: { icon: Wallet, color: 'text-yellow-500', domain: '充电桩运营' },
  emergency_support: { icon: AlertTriangle, color: 'text-red-500', domain: '充电桩运营' },
  data_analyst: { icon: BarChart3, color: 'text-purple-500', domain: '能源管理' },
  maintenance_expert: { icon: Wrench, color: 'text-orange-500', domain: '能源管理' },
  energy_advisor: { icon: Leaf, color: 'text-emerald-500', domain: '能源管理' },
  scheduling_advisor: { icon: Calendar, color: 'text-indigo-500', domain: '能源管理' },
};

const AGENT_DISPLAY_NAMES: Record<string, string> = {
  travel_assistant: '出行助手',
  charging_manager: '充电管家',
  billing_advisor: '费用顾问',
  emergency_support: '故障急救',
  data_analyst: '数据分析师',
  maintenance_expert: '运维专家',
  energy_advisor: '能源顾问',
  scheduling_advisor: '调度顾问',
};

const MultiAgentPanel: React.FC<MultiAgentPanelProps> = ({
  isMultiAgentMode,
  onToggleMultiAgentMode,
  routingMode,
  onRoutingModeChange,
  selectedAgents,
  onSelectedAgentsChange,
}) => {
  const [agents, setAgents] = useState<A2AAgentInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [connectedCount, setConnectedCount] = useState(0);
  const [totalCount, setTotalCount] = useState(0);

  const fetchAgents = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await a2aApi.getAgents();
      setAgents(response.agents);
      setTotalCount(response.total);
      setConnectedCount(response.agents.filter(a => a.connected).length);
    } catch (err) {
      console.error('Failed to fetch agents:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isMultiAgentMode) {
      fetchAgents();
    }
  }, [isMultiAgentMode, fetchAgents]);

  const handleAgentToggle = (agentName: string) => {
    if (selectedAgents.includes(agentName)) {
      onSelectedAgentsChange(selectedAgents.filter((a) => a !== agentName));
    } else {
      onSelectedAgentsChange([...selectedAgents, agentName]);
    }
  };

  const getAgentIcon = (agentName: string) => {
    const meta = AGENT_METADATA[agentName];
    if (meta) {
      const Icon = meta.icon;
      return <Icon className={`w-4 h-4 ${meta.color}`} />;
    }
    return <Users className="w-4 h-4 text-gray-500" />;
  };

  const getAgentDisplayName = (agentName: string) => {
    return AGENT_DISPLAY_NAMES[agentName] || agentName;
  };

  const getAgentDomain = (agentName: string) => {
    return AGENT_METADATA[agentName]?.domain || '未知';
  };

  // Group agents by domain
  const groupedAgents = agents.reduce(
    (acc, agent) => {
      // Use domain from backend, fallback to AGENT_METADATA
      const domain = agent.domain === 'charging' ? '充电桩运营' :
                     agent.domain === 'energy' ? '能源管理' :
                     getAgentDomain(agent.name);
      if (!acc[domain]) {
        acc[domain] = [];
      }
      acc[domain].push(agent);
      return acc;
    },
    {} as Record<string, A2AAgentInfo[]>
  );

  if (!isMultiAgentMode) {
    return (
      <div className="px-4 py-2 bg-gradient-to-r from-indigo-50 to-purple-50 border-b">
        <button
          onClick={() => onToggleMultiAgentMode(true)}
          className="flex items-center gap-2 text-sm text-indigo-600 hover:text-indigo-800 transition-colors"
        >
          <Users className="w-4 h-4" />
          <span>启用多智能体模式 (A2A)</span>
        </button>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border-b">
      {/* Header */}
      <div className="px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Users className="w-5 h-5 text-indigo-600" />
            <span className="font-medium text-indigo-900">多智能体模式</span>
          </div>

          {/* Status indicator */}
          <div className="flex items-center gap-1.5 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                connectedCount === totalCount
                  ? 'bg-green-500'
                  : connectedCount > 0
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
              }`}
            />
            <span className="text-gray-600">
              {connectedCount}/{totalCount} 智能体在线
            </span>
          </div>

          {/* Refresh button */}
          <button
            onClick={fetchAgents}
            disabled={isLoading}
            className="p-1 text-gray-400 hover:text-gray-600 rounded transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        <div className="flex items-center gap-2">
          {/* Routing Mode Selector */}
          <div className="flex items-center gap-1 bg-white rounded-lg p-1 shadow-sm">
            <button
              onClick={() => onRoutingModeChange('auto')}
              className={`px-3 py-1 text-xs rounded-md transition-colors ${
                routingMode === 'auto'
                  ? 'bg-indigo-500 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
              title="自动根据关键词选择智能体"
            >
              <Zap className="w-3 h-3 inline mr-1" />
              自动
            </button>
            <button
              onClick={() => onRoutingModeChange('parallel')}
              className={`px-3 py-1 text-xs rounded-md transition-colors ${
                routingMode === 'parallel'
                  ? 'bg-indigo-500 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
              title="并行调用多个智能体"
            >
              <Users className="w-3 h-3 inline mr-1" />
              并行
            </button>
            <button
              onClick={() => onRoutingModeChange('sequential')}
              className={`px-3 py-1 text-xs rounded-md transition-colors ${
                routingMode === 'sequential'
                  ? 'bg-indigo-500 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
              title="串行调用智能体链"
            >
              <ArrowRight className="w-3 h-3 inline mr-1" />
              串行
            </button>
          </div>

          {/* Expand/Collapse */}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-white rounded transition-colors"
          >
            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {/* Close button */}
          <button
            onClick={() => onToggleMultiAgentMode(false)}
            className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-white rounded transition-colors"
          >
            <XCircle className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Expanded Agent List */}
      {isExpanded && (
        <div className="px-4 pb-3">
          {/* Selected agents for PARALLEL/SEQUENTIAL */}
          {(routingMode === 'parallel' || routingMode === 'sequential') && (
            <div className="mb-2 text-xs text-gray-500">
              {routingMode === 'parallel'
                ? '选择要并行调用的智能体:'
                : '选择智能体链 (按选择顺序执行):'}
            </div>
          )}

          {/* Agent groups */}
          {Object.entries(groupedAgents).map(([domain, domainAgents]) => (
            <div key={domain} className="mb-3">
              <div className="text-xs font-medium text-gray-500 mb-1.5">{domain}</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {domainAgents.map((agent) => {
                  const isSelected = selectedAgents.includes(agent.name);
                  const isConnected = agent.connected;
                  const canSelect = routingMode !== 'auto' && isConnected;

                  return (
                    <button
                      key={agent.name}
                      onClick={() => canSelect && handleAgentToggle(agent.name)}
                      disabled={!canSelect}
                      className={`flex items-center gap-2 p-2 rounded-lg text-left transition-all ${
                        isSelected
                          ? 'bg-indigo-100 border-2 border-indigo-400'
                          : 'bg-white border border-gray-200 hover:border-gray-300'
                      } ${!canSelect ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}`}
                    >
                      <div className="flex-shrink-0">{getAgentIcon(agent.name)}</div>
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-medium text-gray-900 truncate">
                          {getAgentDisplayName(agent.name)}
                        </div>
                        <div className="flex items-center gap-1">
                          {isConnected ? (
                            <CheckCircle className="w-3 h-3 text-green-500" />
                          ) : (
                            <XCircle className="w-3 h-3 text-red-500" />
                          )}
                          <span className="text-xs text-gray-400">
                            {isConnected ? '在线' : '离线'}
                          </span>
                        </div>
                      </div>
                      {isSelected && routingMode === 'sequential' && (
                        <span className="flex-shrink-0 w-5 h-5 bg-indigo-500 text-white text-xs rounded-full flex items-center justify-center">
                          {selectedAgents.indexOf(agent.name) + 1}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}

          {/* Selected agents summary */}
          {(routingMode === 'parallel' || routingMode === 'sequential') &&
            selectedAgents.length > 0 && (
              <div className="mt-2 p-2 bg-white rounded-lg border border-indigo-200">
                <div className="text-xs text-gray-500 mb-1">
                  {routingMode === 'parallel' ? '将并行调用:' : '执行顺序:'}
                </div>
                <div className="flex flex-wrap gap-1">
                  {selectedAgents.map((agent, index) => (
                    <span
                      key={agent}
                      className="inline-flex items-center gap-1 px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded text-xs"
                    >
                      {routingMode === 'sequential' && (
                        <span className="font-medium">{index + 1}.</span>
                      )}
                      {getAgentDisplayName(agent)}
                    </span>
                  ))}
                </div>
              </div>
            )}
        </div>
      )}
    </div>
  );
};

export default MultiAgentPanel;
