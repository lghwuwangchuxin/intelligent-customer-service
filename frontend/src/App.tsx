import React, { useRef, useEffect, useState, useCallback } from 'react';
import {
  Bot,
  Database,
  RefreshCw,
  Info,
  MessageSquare,
  Settings,
  Brain,
  PanelLeftClose,
  PanelLeft,
  Zap,
  User,
  Users,
  Activity,
  BarChart3,
} from 'lucide-react';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import TypingIndicator from './components/TypingIndicator';
import KnowledgePanel from './components/KnowledgePanel';
import AgentPanel from './components/AgentPanel';
import ThinkingIndicator from './components/ThinkingIndicator';
import ModelConfigPanel from './components/ModelConfigPanel';
import ConversationHistory from './components/ConversationHistory';
import MemoryPanel from './components/MemoryPanel';
import MultiAgentPanel from './components/MultiAgentPanel';
import ServiceStatusDashboard from './components/ServiceStatusDashboard';
import MonitoringDashboard from './components/MonitoringDashboard';
import useAgentChat from './hooks/useAgentChat';
import useMultiAgentChat from './hooks/useMultiAgentChat';
import { systemApi, SystemInfo, ModelConfig, conversationApi, A2ARoutingMode } from './services/api';
import { Message } from './components/ChatMessage';

const App: React.FC = () => {
  const [userId, setUserId] = useState<string>(() => {
    // Try to get userId from localStorage or generate a new one
    const stored = localStorage.getItem('user_id');
    if (stored) return stored;
    const newId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem('user_id', newId);
    return newId;
  });

  const {
    messages,
    isLoading,
    error,
    useRag,
    setUseRag,
    agentMode,
    setAgentMode,
    useLangGraph,
    setUseLangGraph,
    conversationId,
    sendMessage,
    clearMessages,
    startNewConversation,
    loadConversation,
    thinkingSteps,
    currentStatus,
    currentStep,
  } = useAgentChat({
    useRag: true,
    agentMode: false,
    useLangGraph: true,
  });

  const [showKnowledgePanel, setShowKnowledgePanel] = useState(false);
  const [showSystemInfo, setShowSystemInfo] = useState(false);
  const [showModelConfig, setShowModelConfig] = useState(false);
  const [showMemoryPanel, setShowMemoryPanel] = useState(false);
  const [showServiceStatus, setShowServiceStatus] = useState(false);
  const [showMonitoring, setShowMonitoring] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Multi-agent mode state
  const [isMultiAgentMode, setIsMultiAgentMode] = useState(false);
  const [routingMode, setRoutingMode] = useState<A2ARoutingMode>('auto');
  const [selectedAgents, setSelectedAgents] = useState<string[]>([]);

  // Multi-agent chat hook
  const {
    messages: multiAgentMessages,
    isLoading: isMultiAgentLoading,
    error: multiAgentError,
    currentAgent,
    currentStatus: multiAgentStatus,
    sendMessage: sendMultiAgentMessage,
    clearMessages: clearMultiAgentMessages,
  } = useMultiAgentChat();

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, multiAgentMessages]);

  // Fetch system info
  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        const info = await systemApi.getInfo();
        setSystemInfo(info);
      } catch (err) {
        console.error('Failed to fetch system info:', err);
      }
    };
    fetchSystemInfo();
  }, []);

  const handleSend = (message: string) => {
    if (isMultiAgentMode) {
      sendMultiAgentMessage(message, routingMode, selectedAgents, false);  // Streaming not supported yet
    } else {
      sendMessage(message, true); // Use streaming by default
    }
  };

  // Get current messages based on mode
  const currentMessages: Message[] = isMultiAgentMode ? multiAgentMessages : messages;
  const currentIsLoading = isMultiAgentMode ? isMultiAgentLoading : isLoading;
  const currentError = isMultiAgentMode ? multiAgentError : error;

  // Handle clear messages based on mode
  const handleClearMessages = () => {
    if (isMultiAgentMode) {
      clearMultiAgentMessages();
    } else {
      clearMessages();
    }
  };


  const handleConfigUpdate = (_config: ModelConfig) => {
    // Refresh system info to reflect new config
    systemApi.getInfo().then(setSystemInfo).catch(console.error);
  };

  const handleSelectConversation = useCallback(async (convId: string) => {
    try {
      const detail = await conversationApi.getDetail(convId);
      const loadedMessages: Message[] = detail.messages.map((m, idx) => ({
        id: `loaded_${idx}_${Date.now()}`,
        role: m.role as 'user' | 'assistant',
        content: m.content,
        timestamp: m.timestamp ? new Date(m.timestamp) : new Date(),
      }));
      loadConversation(convId, loadedMessages);
    } catch (err) {
      console.error('Failed to load conversation:', err);
    }
  }, [loadConversation]);

  const handleNewConversation = useCallback(() => {
    startNewConversation();
  }, [startNewConversation]);

  const handleDeleteConversation = useCallback((convId: string) => {
    if (conversationId === convId) {
      startNewConversation();
    }
  }, [conversationId, startNewConversation]);

  const handleUserIdChange = useCallback((newUserId: string) => {
    setUserId(newUserId);
    localStorage.setItem('user_id', newUserId);
  }, []);

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      {showSidebar && (
        <div className="w-72 flex-shrink-0 border-r bg-white flex flex-col">
          {/* Sidebar Header */}
          <div className="p-3 border-b flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Bot className="w-6 h-6 text-blue-500" />
              <span className="font-semibold text-gray-900">智能客服</span>
            </div>
            <button
              onClick={() => setShowSidebar(false)}
              className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded"
            >
              <PanelLeftClose size={18} />
            </button>
          </div>

          {/* User Info */}
          <div className="p-3 border-b bg-gray-50">
            <div className="flex items-center gap-2 text-sm">
              <User size={14} className="text-gray-400" />
              <span className="text-gray-600 truncate" title={userId}>
                {userId.substring(0, 20)}...
              </span>
            </div>
          </div>

          {/* Conversation History */}
          <div className="flex-1 overflow-hidden">
            <ConversationHistory
              currentConversationId={conversationId}
              onSelectConversation={handleSelectConversation}
              onNewConversation={handleNewConversation}
              onDeleteConversation={handleDeleteConversation}
            />
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white shadow-sm border-b border-gray-200">
          <div className="px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              {!showSidebar && (
                <button
                  onClick={() => setShowSidebar(true)}
                  className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
                >
                  <PanelLeft size={20} />
                </button>
              )}
              <div>
                <h1 className="text-lg font-semibold text-gray-900">
                  {conversationId ? '对话中' : '新对话'}
                </h1>
                <p className="text-xs text-gray-500">
                  {systemInfo
                    ? `${systemInfo.llm_info.provider} / ${systemInfo.llm_info.model}`
                    : '连接中...'}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {/* Mode indicators */}
              {isMultiAgentMode && (
                <div className="flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-indigo-100 text-indigo-700">
                  <Users className="w-3 h-3" />
                  多智能体 ({routingMode})
                </div>
              )}

              {!isMultiAgentMode && agentMode && (
                <div className="flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-purple-100 text-purple-700">
                  <Zap className="w-3 h-3" />
                  {useLangGraph ? 'LangGraph' : 'ReAct'}
                </div>
              )}

              {!isMultiAgentMode && !agentMode && (
                <div
                  className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs ${
                    useRag
                      ? 'bg-green-100 text-green-700'
                      : 'bg-gray-100 text-gray-600'
                  }`}
                >
                  <Database className="w-3 h-3" />
                  {useRag ? '知识库已启用' : '知识库已禁用'}
                </div>
              )}

              {/* LangGraph Toggle (when in agent mode and not multi-agent) */}
              {!isMultiAgentMode && agentMode && (
                <button
                  onClick={() => setUseLangGraph(!useLangGraph)}
                  className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs transition-colors ${
                    useLangGraph
                      ? 'bg-purple-100 text-purple-700'
                      : 'bg-gray-100 text-gray-600'
                  }`}
                  title={useLangGraph ? '使用 LangGraph Agent' : '使用 ReAct Agent'}
                >
                  <Zap className="w-3 h-3" />
                  {useLangGraph ? 'LangGraph' : 'ReAct'}
                </button>
              )}

              {/* Service Status button */}
              <button
                onClick={() => setShowServiceStatus(true)}
                className="p-2 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                title="服务状态"
              >
                <Activity className="w-5 h-5" />
              </button>

              {/* Monitoring button */}
              <button
                onClick={() => setShowMonitoring(true)}
                className="p-2 text-gray-500 hover:text-orange-600 hover:bg-orange-50 rounded-lg transition-colors"
                title="系统监控"
              >
                <BarChart3 className="w-5 h-5" />
              </button>

              {/* Memory Panel button */}
              <button
                onClick={() => setShowMemoryPanel(true)}
                className="p-2 text-gray-500 hover:text-purple-600 hover:bg-purple-50 rounded-lg transition-colors"
                title="长期记忆"
              >
                <Brain className="w-5 h-5" />
              </button>

              {/* Model Config button */}
              <button
                onClick={() => setShowModelConfig(true)}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                title="模型配置"
              >
                <Settings className="w-5 h-5" />
              </button>

              {/* Knowledge Base button */}
              <button
                onClick={() => setShowKnowledgePanel(true)}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                title="知识库管理"
              >
                <Database className="w-5 h-5" />
              </button>

              {/* System Info button */}
              <button
                onClick={() => setShowSystemInfo(!showSystemInfo)}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                title="系统信息"
              >
                <Info className="w-5 h-5" />
              </button>

              {/* Clear chat button */}
              <button
                onClick={handleClearMessages}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                title="清空对话"
              >
                <RefreshCw className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* System Info Panel */}
          {showSystemInfo && systemInfo && (
            <div className="px-4 pb-3">
              <div className="bg-gray-50 rounded-lg p-3 text-sm">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-gray-600">
                  <div>
                    <span className="text-gray-400">应用名称:</span> {systemInfo.app_name}
                  </div>
                  <div>
                    <span className="text-gray-400">版本:</span> {systemInfo.version}
                  </div>
                  <div>
                    <span className="text-gray-400">LLM提供商:</span>{' '}
                    {systemInfo.llm_info.provider}
                  </div>
                  <div>
                    <span className="text-gray-400">模型:</span> {systemInfo.llm_info.model}
                  </div>
                  <div>
                    <span className="text-gray-400">温度:</span>{' '}
                    {systemInfo.llm_info.temperature}
                  </div>
                  <div>
                    <span className="text-gray-400">状态:</span>{' '}
                    <span className="text-green-600">{systemInfo.status}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </header>

        {/* Agent Panel (only show when not in multi-agent mode) */}
        {!isMultiAgentMode && (
          <AgentPanel
            isAgentMode={agentMode}
            onToggleAgentMode={setAgentMode}
            currentStep={currentStep}
            status={currentStatus}
          />
        )}

        {/* Multi-Agent Panel */}
        <MultiAgentPanel
          isMultiAgentMode={isMultiAgentMode}
          onToggleMultiAgentMode={(enabled) => {
            setIsMultiAgentMode(enabled);
            if (enabled) {
              // Disable agent mode when enabling multi-agent mode
              setAgentMode(false);
            }
          }}
          routingMode={routingMode}
          onRoutingModeChange={setRoutingMode}
          selectedAgents={selectedAgents}
          onSelectedAgentsChange={setSelectedAgents}
        />

        {/* Main Chat Area */}
        <main className="flex-1 overflow-hidden">
          <div className="h-full max-w-4xl mx-auto flex flex-col">
            {/* Messages */}
            <div className="flex-1 overflow-y-auto">
              {currentMessages.length === 0 ? (
                /* Welcome screen */
                <div className="h-full flex flex-col items-center justify-center text-center p-8">
                  <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-4 ${
                    isMultiAgentMode ? 'bg-indigo-100' : 'bg-primary-100'
                  }`}>
                    {isMultiAgentMode ? (
                      <Users className="w-8 h-8 text-indigo-500" />
                    ) : (
                      <MessageSquare className="w-8 h-8 text-primary-500" />
                    )}
                  </div>
                  <h2 className="text-xl font-semibold text-gray-900 mb-2">
                    {isMultiAgentMode ? '多智能体协作模式' : '欢迎使用智能客服'}
                  </h2>
                  <p className="text-gray-500 max-w-md mb-6">
                    {isMultiAgentMode
                      ? routingMode === 'auto'
                        ? '系统将根据您的问题自动选择最合适的智能体为您服务。'
                        : routingMode === 'parallel'
                        ? '您选择的多个智能体将同时处理您的问题，提供多角度的解答。'
                        : '问题将依次经过多个智能体处理，形成完整的分析链。'
                      : agentMode
                      ? useLangGraph
                        ? '当前为 LangGraph Agent 模式，支持任务规划和并行工具执行。'
                        : '当前为 ReAct Agent 模式，我可以使用工具来帮助您解决复杂问题。'
                      : useRag
                      ? '我会根据知识库中的内容为您提供准确的答案。'
                      : '目前使用通用对话模式。'}
                  </p>
                  <div className="space-y-2 text-sm text-gray-400">
                    <p>您可以问我：</p>
                    <div className="flex flex-wrap gap-2 justify-center">
                      {isMultiAgentMode ? (
                        ['附近有什么充电站？', '充电中断怎么办？', '生成能效报告', '设备健康检查'].map(
                          (topic) => (
                            <span
                              key={topic}
                              onClick={() => handleSend(topic)}
                              className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full cursor-pointer hover:bg-indigo-200 transition-colors"
                            >
                              {topic}
                            </span>
                          )
                        )
                      ) : (
                        ['产品使用问题', '服务咨询', '技术支持', '常见问题'].map(
                          (topic) => (
                            <span
                              key={topic}
                              onClick={() => handleSend(`我想了解${topic}`)}
                              className="px-3 py-1 bg-gray-100 rounded-full cursor-pointer hover:bg-gray-200 transition-colors"
                            >
                              {topic}
                            </span>
                          )
                        )
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                /* Message list */
                <div className="divide-y divide-gray-100">
                  {currentMessages.map((message) => (
                    <ChatMessage key={message.id} message={message} />
                  ))}
                  {currentIsLoading && !agentMode && !isMultiAgentMode && <TypingIndicator />}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Agent Thinking Indicator */}
            {!isMultiAgentMode && agentMode && isLoading && (
              <ThinkingIndicator
                steps={thinkingSteps}
                currentStatus={currentStatus}
                isLoading={isLoading}
              />
            )}

            {/* Multi-Agent Status Indicator */}
            {isMultiAgentMode && isMultiAgentLoading && (
              <div className="px-4 py-3 bg-indigo-50 border-t border-indigo-100">
                <div className="flex items-center gap-3">
                  <div className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                  <div>
                    <div className="text-sm font-medium text-indigo-700">
                      {multiAgentStatus || '正在处理...'}
                    </div>
                    {currentAgent && (
                      <div className="text-xs text-indigo-500">
                        当前智能体: {currentAgent}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Error display */}
            {currentError && (
              <div className="px-4 py-2 bg-red-50 text-red-600 text-sm">
                {currentError}
              </div>
            )}

            {/* Input */}
            <ChatInput
              onSend={handleSend}
              disabled={currentIsLoading}
              useRag={useRag}
              onToggleRag={setUseRag}
            />
          </div>
        </main>
      </div>

      {/* Memory Panel Modal */}
      {showMemoryPanel && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-md h-[600px] flex flex-col">
            <div className="p-4 border-b flex items-center justify-between">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <Brain className="text-purple-500" />
                长期记忆管理
              </h2>
              <button
                onClick={() => setShowMemoryPanel(false)}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
              >
                ×
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <MemoryPanel
                userId={userId}
                onUserIdChange={handleUserIdChange}
              />
            </div>
          </div>
        </div>
      )}

      {/* Knowledge Panel Modal */}
      <KnowledgePanel
        isOpen={showKnowledgePanel}
        onClose={() => setShowKnowledgePanel(false)}
      />

      {/* Model Config Panel Modal */}
      <ModelConfigPanel
        isOpen={showModelConfig}
        onClose={() => setShowModelConfig(false)}
        onConfigUpdate={handleConfigUpdate}
      />

      {/* Service Status Dashboard Modal */}
      <ServiceStatusDashboard
        isOpen={showServiceStatus}
        onClose={() => setShowServiceStatus(false)}
      />

      {/* Monitoring Dashboard Modal */}
      <MonitoringDashboard
        isOpen={showMonitoring}
        onClose={() => setShowMonitoring(false)}
      />
    </div>
  );
};

export default App;
