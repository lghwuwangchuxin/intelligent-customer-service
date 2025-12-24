import React, { useRef, useEffect, useState } from 'react';
import { Bot, Database, RefreshCw, Info, MessageSquare, Settings } from 'lucide-react';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import TypingIndicator from './components/TypingIndicator';
import KnowledgePanel from './components/KnowledgePanel';
import AgentPanel from './components/AgentPanel';
import ThinkingIndicator from './components/ThinkingIndicator';
import ModelConfigPanel from './components/ModelConfigPanel';
import useAgentChat from './hooks/useAgentChat';
import { systemApi, SystemInfo, ModelConfig } from './services/api';

const App: React.FC = () => {
  const {
    messages,
    isLoading,
    error,
    useRag,
    setUseRag,
    agentMode,
    setAgentMode,
    sendMessage,
    clearMessages,
    thinkingSteps,
    currentStatus,
    currentStep,
  } = useAgentChat({ useRag: true, agentMode: false });

  const [showKnowledgePanel, setShowKnowledgePanel] = useState(false);
  const [showSystemInfo, setShowSystemInfo] = useState(false);
  const [showModelConfig, setShowModelConfig] = useState(false);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

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
    sendMessage(message, true); // Use streaming by default
  };

  const handleConfigUpdate = (config: ModelConfig) => {
    // Refresh system info to reflect new config
    systemApi.getInfo().then(setSystemInfo).catch(console.error);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-primary-500 rounded-lg flex items-center justify-center">
              <Bot className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-900">智能客服</h1>
              <p className="text-xs text-gray-500">
                {systemInfo
                  ? `${systemInfo.llm_info.provider} / ${systemInfo.llm_info.model}`
                  : '连接中...'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Mode indicators */}
            {!agentMode && (
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
              onClick={clearMessages}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              title="清空对话"
            >
              <RefreshCw className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* System Info Panel */}
        {showSystemInfo && systemInfo && (
          <div className="max-w-4xl mx-auto px-4 pb-3">
            <div className="bg-gray-50 rounded-lg p-3 text-sm">
              <div className="grid grid-cols-2 gap-2 text-gray-600">
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

      {/* Agent Panel */}
      <AgentPanel
        isAgentMode={agentMode}
        onToggleAgentMode={setAgentMode}
        currentStep={currentStep}
        status={currentStatus}
      />

      {/* Main Chat Area */}
      <main className="flex-1 overflow-hidden">
        <div className="h-full max-w-4xl mx-auto flex flex-col">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto">
            {messages.length === 0 ? (
              /* Welcome screen */
              <div className="h-full flex flex-col items-center justify-center text-center p-8">
                <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mb-4">
                  <MessageSquare className="w-8 h-8 text-primary-500" />
                </div>
                <h2 className="text-xl font-semibold text-gray-900 mb-2">
                  欢迎使用智能客服
                </h2>
                <p className="text-gray-500 max-w-md mb-6">
                  我是您的智能客服助手，可以回答您的问题。
                  {agentMode
                    ? '当前为 Agent 模式，我可以使用工具来帮助您解决复杂问题。'
                    : useRag
                    ? '我会根据知识库中的内容为您提供准确的答案。'
                    : '目前使用通用对话模式。'}
                </p>
                <div className="space-y-2 text-sm text-gray-400">
                  <p>您可以问我：</p>
                  <div className="flex flex-wrap gap-2 justify-center">
                    {['产品使用问题', '服务咨询', '技术支持', '常见问题'].map(
                      (topic) => (
                        <span
                          key={topic}
                          onClick={() => handleSend(`我想了解${topic}`)}
                          className="px-3 py-1 bg-gray-100 rounded-full cursor-pointer hover:bg-gray-200 transition-colors"
                        >
                          {topic}
                        </span>
                      )
                    )}
                  </div>
                </div>
              </div>
            ) : (
              /* Message list */
              <div className="divide-y divide-gray-100">
                {messages.map((message) => (
                  <ChatMessage key={message.id} message={message} />
                ))}
                {isLoading && !agentMode && <TypingIndicator />}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Agent Thinking Indicator */}
          {agentMode && isLoading && (
            <ThinkingIndicator
              steps={thinkingSteps}
              currentStatus={currentStatus}
              isLoading={isLoading}
            />
          )}

          {/* Error display */}
          {error && (
            <div className="px-4 py-2 bg-red-50 text-red-600 text-sm">
              {error}
            </div>
          )}

          {/* Input */}
          <ChatInput
            onSend={handleSend}
            disabled={isLoading}
            useRag={useRag}
            onToggleRag={setUseRag}
          />
        </div>
      </main>

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
    </div>
  );
};

export default App;
