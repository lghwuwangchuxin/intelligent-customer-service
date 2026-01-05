import React, { useState, useEffect, useCallback } from 'react';
import {
  Activity,
  BarChart3,
  Clock,
  AlertTriangle,
  TrendingUp,
  RefreshCw,
  X,
  Zap,
  MessageSquare,
  Timer,
} from 'lucide-react';
import { monitoringApi, MonitoringStats, MetricData, TraceSpan } from '../services/api';

interface MonitoringDashboardProps {
  isOpen: boolean;
  onClose: () => void;
}

type TabType = 'overview' | 'metrics' | 'traces';

const formatDuration = (ms: number): string => {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
};

const formatTimestamp = (timestamp: string): string => {
  try {
    return new Date(timestamp).toLocaleTimeString('zh-CN');
  } catch {
    return timestamp;
  }
};

export const MonitoringDashboard: React.FC<MonitoringDashboardProps> = ({
  isOpen,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [stats, setStats] = useState<MonitoringStats | null>(null);
  const [metrics, setMetrics] = useState<MetricData[]>([]);
  const [traces, setTraces] = useState<TraceSpan[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchStats = useCallback(async () => {
    try {
      const data = await monitoringApi.getStats();
      setStats(data);
    } catch (err) {
      console.error('Failed to fetch monitoring stats:', err);
    }
  }, []);

  const fetchMetrics = useCallback(async () => {
    try {
      const data = await monitoringApi.getMetrics();
      setMetrics(data);
    } catch (err) {
      console.error('Failed to fetch metrics:', err);
    }
  }, []);

  const fetchTraces = useCallback(async () => {
    try {
      const data = await monitoringApi.getTraces(20);
      setTraces(data);
    } catch (err) {
      console.error('Failed to fetch traces:', err);
    }
  }, []);

  const refreshAll = useCallback(async () => {
    setIsLoading(true);
    await Promise.all([fetchStats(), fetchMetrics(), fetchTraces()]);
    setLastUpdated(new Date());
    setIsLoading(false);
  }, [fetchStats, fetchMetrics, fetchTraces]);

  useEffect(() => {
    if (isOpen) {
      refreshAll();
    }
  }, [isOpen, refreshAll]);

  useEffect(() => {
    if (!autoRefresh || !isOpen) return;
    const interval = setInterval(refreshAll, 30000);
    return () => clearInterval(interval);
  }, [autoRefresh, isOpen, refreshAll]);

  if (!isOpen) return null;

  const tabs: { id: TabType; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: '概览', icon: <BarChart3 size={16} /> },
    { id: 'metrics', label: '指标', icon: <TrendingUp size={16} /> },
    { id: 'traces', label: '追踪', icon: <Activity size={16} /> },
  ];

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="p-4 border-b flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-orange-100 flex items-center justify-center">
              <Activity className="text-orange-600" size={24} />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">系统监控</h2>
              <p className="text-sm text-gray-500">
                {lastUpdated
                  ? `最后更新: ${lastUpdated.toLocaleTimeString()}`
                  : '加载中...'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`p-2 rounded-lg transition-colors ${
                autoRefresh ? 'bg-orange-100 text-orange-600' : 'text-gray-500 hover:bg-gray-100'
              }`}
              title={autoRefresh ? '自动刷新已开启' : '开启自动刷新'}
            >
              <Zap size={18} />
            </button>
            <button
              onClick={refreshAll}
              disabled={isLoading}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
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

        {/* Tabs */}
        <div className="px-4 border-b">
          <div className="flex gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'text-orange-600 border-orange-500'
                    : 'text-gray-500 border-transparent hover:text-gray-700'
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {/* Overview Tab */}
          {activeTab === 'overview' && (
            <div className="space-y-4">
              {/* Stats Cards */}
              {stats ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-blue-50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <MessageSquare className="text-blue-500" size={20} />
                      <span className="text-xs text-blue-600">请求数</span>
                    </div>
                    <p className="text-2xl font-bold text-blue-700">{stats.total_requests}</p>
                    <p className="text-xs text-blue-500 mt-1">
                      {stats.requests_per_minute.toFixed(1)}/min
                    </p>
                  </div>

                  <div className="bg-green-50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <Timer className="text-green-500" size={20} />
                      <span className="text-xs text-green-600">平均延迟</span>
                    </div>
                    <p className="text-2xl font-bold text-green-700">
                      {formatDuration(stats.avg_latency_ms)}
                    </p>
                    <p className="text-xs text-green-500 mt-1">响应时间</p>
                  </div>

                  <div className="bg-red-50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <AlertTriangle className="text-red-500" size={20} />
                      <span className="text-xs text-red-600">错误率</span>
                    </div>
                    <p className="text-2xl font-bold text-red-700">
                      {(stats.error_rate * 100).toFixed(2)}%
                    </p>
                    <p className="text-xs text-red-500 mt-1">失败请求</p>
                  </div>

                  <div className="bg-purple-50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <Activity className="text-purple-500" size={20} />
                      <span className="text-xs text-purple-600">活跃对话</span>
                    </div>
                    <p className="text-2xl font-bold text-purple-700">
                      {stats.active_conversations}
                    </p>
                    <p className="text-xs text-purple-500 mt-1">进行中</p>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[1, 2, 3, 4].map((i) => (
                    <div key={i} className="bg-gray-50 rounded-lg p-4 animate-pulse">
                      <div className="h-20"></div>
                    </div>
                  ))}
                </div>
              )}

              {/* Quick Info */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-900 mb-3 flex items-center gap-2">
                  <Clock size={16} />
                  系统状态
                </h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">服务状态</span>
                    <span className="text-green-600 font-medium">正常运行</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">监控模式</span>
                    <span className="text-gray-900">{autoRefresh ? '自动刷新' : '手动刷新'}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Metrics Tab */}
          {activeTab === 'metrics' && (
            <div className="space-y-3">
              {metrics.length === 0 ? (
                <div className="text-center py-12 text-gray-500">
                  <TrendingUp size={40} className="mx-auto mb-2 opacity-50" />
                  <p>暂无指标数据</p>
                  <p className="text-sm text-gray-400 mt-1">
                    监控服务可能未启用或尚未收集到数据
                  </p>
                </div>
              ) : (
                metrics.map((metric, index) => (
                  <div key={index} className="bg-white border rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium text-gray-900">{metric.name}</h4>
                        {metric.labels && Object.keys(metric.labels).length > 0 && (
                          <div className="flex gap-2 mt-1">
                            {Object.entries(metric.labels).map(([key, value]) => (
                              <span
                                key={key}
                                className="text-xs px-2 py-0.5 bg-gray-100 rounded"
                              >
                                {key}: {value}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="text-right">
                        <p className="text-xl font-bold text-gray-900">
                          {metric.value.toFixed(2)}
                          {metric.unit && (
                            <span className="text-sm font-normal text-gray-500 ml-1">
                              {metric.unit}
                            </span>
                          )}
                        </p>
                        <p className="text-xs text-gray-400">{formatTimestamp(metric.timestamp)}</p>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {/* Traces Tab */}
          {activeTab === 'traces' && (
            <div className="space-y-2">
              {traces.length === 0 ? (
                <div className="text-center py-12 text-gray-500">
                  <Activity size={40} className="mx-auto mb-2 opacity-50" />
                  <p>暂无追踪数据</p>
                  <p className="text-sm text-gray-400 mt-1">
                    监控服务可能未启用或尚未收集到数据
                  </p>
                </div>
              ) : (
                traces.map((trace, index) => (
                  <div
                    key={index}
                    className={`bg-white border rounded-lg p-3 ${
                      trace.status === 'error' ? 'border-red-200 bg-red-50' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span
                          className={`w-2 h-2 rounded-full ${
                            trace.status === 'ok'
                              ? 'bg-green-500'
                              : trace.status === 'error'
                              ? 'bg-red-500'
                              : 'bg-yellow-500'
                          }`}
                        />
                        <span className="font-medium text-gray-900 text-sm">{trace.name}</span>
                      </div>
                      <div className="flex items-center gap-3 text-xs text-gray-500">
                        {trace.duration_ms && (
                          <span>{formatDuration(trace.duration_ms)}</span>
                        )}
                        <span>{formatTimestamp(trace.start_time)}</span>
                      </div>
                    </div>
                    <div className="mt-1 text-xs text-gray-400">
                      <span className="font-mono">{trace.trace_id.slice(0, 16)}...</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MonitoringDashboard;
