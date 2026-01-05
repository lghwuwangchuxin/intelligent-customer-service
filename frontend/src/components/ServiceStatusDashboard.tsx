import React, { useState, useEffect, useCallback } from 'react';
import {
  Activity,
  Server,
  CheckCircle,
  XCircle,
  AlertCircle,
  RefreshCw,
  Cpu,
  Database,
  Brain,
  Bot,
  Users,
  Wrench,
  BarChart3,
  Zap,
  X,
} from 'lucide-react';
import { servicesApi, SERVICES, ServiceHealth, AllServicesHealth } from '../services/api';

interface ServiceStatusDashboardProps {
  isOpen: boolean;
  onClose: () => void;
}

// Service icon mapping
const SERVICE_ICONS: Record<string, React.ReactNode> = {
  'api-gateway': <Server size={18} />,
  'mcp-service': <Wrench size={18} />,
  'rag-service': <Database size={18} />,
  'evaluation-service': <BarChart3 size={18} />,
  'monitoring-service': <Activity size={18} />,
  'single-agent-service': <Bot size={18} />,
  'multi-agent-service': <Users size={18} />,
  'llm-manager-service': <Cpu size={18} />,
  'memory-service': <Brain size={18} />,
};

// Service display names
const SERVICE_NAMES: Record<string, string> = {
  'api-gateway': 'API 网关',
  'mcp-service': 'MCP 工具服务',
  'rag-service': 'RAG 检索服务',
  'evaluation-service': '评估服务',
  'monitoring-service': '监控服务',
  'single-agent-service': '单智能体服务',
  'multi-agent-service': '多智能体服务',
  'llm-manager-service': 'LLM 管理服务',
  'memory-service': '记忆管理服务',
};

// Service descriptions
const SERVICE_DESCRIPTIONS: Record<string, string> = {
  'api-gateway': '统一API入口，请求路由和认证',
  'mcp-service': 'Model Context Protocol工具集成',
  'rag-service': '知识库检索增强生成',
  'evaluation-service': 'RAG和对话质量评估',
  'monitoring-service': '系统监控和指标收集',
  'single-agent-service': '单智能体对话处理',
  'multi-agent-service': '多智能体协作路由',
  'llm-manager-service': '大语言模型统一管理',
  'memory-service': '对话记忆和用户偏好管理',
};

const StatusIcon: React.FC<{ status: ServiceHealth['status'] }> = ({ status }) => {
  switch (status) {
    case 'healthy':
      return <CheckCircle size={18} className="text-green-500" />;
    case 'unhealthy':
      return <XCircle size={18} className="text-red-500" />;
    default:
      return <AlertCircle size={18} className="text-yellow-500" />;
  }
};

const StatusBadge: React.FC<{ status: AllServicesHealth['overall'] }> = ({ status }) => {
  const configs = {
    healthy: { bg: 'bg-green-100', text: 'text-green-700', label: '全部正常' },
    degraded: { bg: 'bg-yellow-100', text: 'text-yellow-700', label: '部分降级' },
    unhealthy: { bg: 'bg-red-100', text: 'text-red-700', label: '服务异常' },
  };
  const config = configs[status];
  return (
    <span className={`px-3 py-1 rounded-full text-sm font-medium ${config.bg} ${config.text}`}>
      {config.label}
    </span>
  );
};

export const ServiceStatusDashboard: React.FC<ServiceStatusDashboardProps> = ({
  isOpen,
  onClose,
}) => {
  const [healthData, setHealthData] = useState<AllServicesHealth | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(false);

  const fetchHealth = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await servicesApi.checkAllHealth();
      setHealthData(data);
      setLastUpdated(new Date());
    } catch (err) {
      console.error('Failed to fetch service health:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      fetchHealth();
    }
  }, [isOpen, fetchHealth]);

  useEffect(() => {
    if (!autoRefresh || !isOpen) return;
    const interval = setInterval(fetchHealth, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [autoRefresh, isOpen, fetchHealth]);

  if (!isOpen) return null;

  const serviceList = Object.values(SERVICES);
  const healthyCount = healthData
    ? Object.values(healthData.services).filter(s => s.status === 'healthy').length
    : 0;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="p-4 border-b flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center">
              <Activity className="text-blue-600" size={24} />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">服务状态监控</h2>
              <p className="text-sm text-gray-500">
                {healthData ? (
                  <>
                    {healthyCount}/{serviceList.length} 服务正常运行
                  </>
                ) : (
                  '正在加载...'
                )}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {healthData && <StatusBadge status={healthData.overall} />}
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`p-2 rounded-lg transition-colors ${
                autoRefresh ? 'bg-blue-100 text-blue-600' : 'text-gray-500 hover:bg-gray-100'
              }`}
              title={autoRefresh ? '自动刷新已开启' : '开启自动刷新'}
            >
              <Zap size={18} />
            </button>
            <button
              onClick={fetchHealth}
              disabled={isLoading}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              title="刷新状态"
            >
              <RefreshCw size={18} className={isLoading ? 'animate-spin' : ''} />
            </button>
            <button
              onClick={onClose}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {/* Overview Stats */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="bg-green-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <span className="text-green-600 font-semibold text-2xl">{healthyCount}</span>
                <CheckCircle className="text-green-500" size={24} />
              </div>
              <p className="text-sm text-green-700 mt-1">正常运行</p>
            </div>
            <div className="bg-red-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <span className="text-red-600 font-semibold text-2xl">
                  {healthData
                    ? Object.values(healthData.services).filter(s => s.status === 'unhealthy').length
                    : 0}
                </span>
                <XCircle className="text-red-500" size={24} />
              </div>
              <p className="text-sm text-red-700 mt-1">服务异常</p>
            </div>
            <div className="bg-yellow-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <span className="text-yellow-600 font-semibold text-2xl">
                  {healthData
                    ? Object.values(healthData.services).filter(s => s.status === 'unknown').length
                    : 0}
                </span>
                <AlertCircle className="text-yellow-500" size={24} />
              </div>
              <p className="text-sm text-yellow-700 mt-1">状态未知</p>
            </div>
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <span className="text-blue-600 font-semibold text-2xl">{serviceList.length}</span>
                <Server className="text-blue-500" size={24} />
              </div>
              <p className="text-sm text-blue-700 mt-1">总服务数</p>
            </div>
          </div>

          {/* Service List */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-gray-700 mb-3">服务详情</h3>
            {serviceList.map((service) => {
              const health = healthData?.services[service.name];
              return (
                <div
                  key={service.name}
                  className={`p-4 rounded-lg border transition-colors ${
                    health?.status === 'healthy'
                      ? 'border-green-200 bg-green-50/50'
                      : health?.status === 'unhealthy'
                      ? 'border-red-200 bg-red-50/50'
                      : 'border-gray-200 bg-gray-50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div
                        className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                          health?.status === 'healthy'
                            ? 'bg-green-100 text-green-600'
                            : health?.status === 'unhealthy'
                            ? 'bg-red-100 text-red-600'
                            : 'bg-gray-100 text-gray-600'
                        }`}
                      >
                        {SERVICE_ICONS[service.name] || <Server size={18} />}
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h4 className="font-medium text-gray-900">
                            {SERVICE_NAMES[service.name] || service.name}
                          </h4>
                          <span className="text-xs px-2 py-0.5 bg-gray-100 rounded text-gray-500">
                            :{service.port}
                          </span>
                        </div>
                        <p className="text-sm text-gray-500">
                          {SERVICE_DESCRIPTIONS[service.name] || service.path}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      {health?.message && (
                        <span className="text-sm text-gray-500 max-w-xs truncate">
                          {health.message}
                        </span>
                      )}
                      <StatusIcon status={health?.status || 'unknown'} />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t bg-gray-50 flex items-center justify-between text-sm text-gray-500">
          <div>
            {lastUpdated && (
              <span>
                最后更新: {lastUpdated.toLocaleTimeString()}
                {autoRefresh && <span className="ml-2 text-blue-500">(自动刷新中)</span>}
              </span>
            )}
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1">
              <CheckCircle size={14} className="text-green-500" />
              <span>正常</span>
            </div>
            <div className="flex items-center gap-1">
              <XCircle size={14} className="text-red-500" />
              <span>异常</span>
            </div>
            <div className="flex items-center gap-1">
              <AlertCircle size={14} className="text-yellow-500" />
              <span>未知</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ServiceStatusDashboard;
