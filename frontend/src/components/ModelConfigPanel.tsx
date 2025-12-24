import React, { useState, useEffect } from 'react';
import { X, Settings, ChevronDown, Check, AlertCircle, Loader2, Play, Eye, EyeOff } from 'lucide-react';
import { configApi, ProviderInfo, ModelConfig, ModelConfigRequest } from '../services/api';

interface ModelConfigPanelProps {
  isOpen: boolean;
  onClose: () => void;
  onConfigUpdate?: (config: ModelConfig) => void;
}

const ModelConfigPanel: React.FC<ModelConfigPanelProps> = ({
  isOpen,
  onClose,
  onConfigUpdate,
}) => {
  // State
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [currentConfig, setCurrentConfig] = useState<ModelConfig | null>(null);
  const [selectedProvider, setSelectedProvider] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [apiKey, setApiKey] = useState<string>('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [customBaseUrl, setCustomBaseUrl] = useState<string>('');
  const [temperature, setTemperature] = useState<number>(0.7);
  const [maxTokens, setMaxTokens] = useState<number>(2048);

  const [isLoading, setIsLoading] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);

  // Get current provider info
  const currentProvider = providers.find(p => p.id === selectedProvider);
  const requiresApiKey = currentProvider?.requires_api_key ?? false;

  // Load providers and current config on mount
  useEffect(() => {
    if (isOpen) {
      loadData();
    }
  }, [isOpen]);

  const loadData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [providersData, configData] = await Promise.all([
        configApi.getProviders(),
        configApi.getCurrentConfig(),
      ]);
      setProviders(providersData);
      setCurrentConfig(configData);

      // Initialize form with current config
      setSelectedProvider(configData.provider);
      setSelectedModel(configData.model);
      setTemperature(configData.temperature);
      setMaxTokens(configData.max_tokens);
      if (configData.base_url) {
        const provider = providersData.find(p => p.id === configData.provider);
        if (provider && configData.base_url !== provider.base_url) {
          setCustomBaseUrl(configData.base_url);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载配置失败');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle provider change
  const handleProviderChange = (providerId: string) => {
    setSelectedProvider(providerId);
    const provider = providers.find(p => p.id === providerId);
    if (provider && provider.models.length > 0) {
      setSelectedModel(provider.models[0]);
    }
    setApiKey('');
    setCustomBaseUrl('');
    setTestResult(null);
    setError(null);
    setSuccess(null);
  };

  // Test configuration
  const handleTest = async () => {
    if (!selectedProvider || !selectedModel) {
      setError('请选择提供商和模型');
      return;
    }

    if (requiresApiKey && !apiKey) {
      setError('此提供商需要 API Key');
      return;
    }

    setIsTesting(true);
    setTestResult(null);
    setError(null);

    try {
      const request: ModelConfigRequest = {
        provider: selectedProvider,
        model: selectedModel,
        api_key: apiKey || undefined,
        base_url: customBaseUrl || undefined,
        temperature,
        max_tokens: 100, // Use small max_tokens for testing
      };

      const result = await configApi.testConfig(request);

      if (result.success) {
        setTestResult({ success: true, message: `连接成功！响应: ${result.response}` });
      } else {
        setTestResult({ success: false, message: result.error || '测试失败' });
      }
    } catch (err) {
      setTestResult({
        success: false,
        message: err instanceof Error ? err.message : '测试连接失败'
      });
    } finally {
      setIsTesting(false);
    }
  };

  // Save configuration
  const handleSave = async () => {
    if (!selectedProvider || !selectedModel) {
      setError('请选择提供商和模型');
      return;
    }

    if (requiresApiKey && !apiKey) {
      setError('此提供商需要 API Key');
      return;
    }

    setIsSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const request: ModelConfigRequest = {
        provider: selectedProvider,
        model: selectedModel,
        api_key: apiKey || undefined,
        base_url: customBaseUrl || undefined,
        temperature,
        max_tokens: maxTokens,
      };

      const newConfig = await configApi.updateConfig(request);
      setCurrentConfig(newConfig);
      setSuccess('配置已保存');
      onConfigUpdate?.(newConfig);

      // Close panel after short delay
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : '保存配置失败');
    } finally {
      setIsSaving(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
              <Settings className="w-5 h-5 text-primary-600" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">模型配置</h2>
              <p className="text-sm text-gray-500">配置 LLM 提供商和模型参数</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
              <span className="ml-3 text-gray-500">加载配置中...</span>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Current Config Info */}
              {currentConfig && (
                <div className="bg-gray-50 rounded-lg p-4">
                  <h3 className="text-sm font-medium text-gray-700 mb-2">当前配置</h3>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-gray-400">提供商:</span>{' '}
                      <span className="text-gray-700">{currentConfig.provider}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">模型:</span>{' '}
                      <span className="text-gray-700">{currentConfig.model}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">温度:</span>{' '}
                      <span className="text-gray-700">{currentConfig.temperature}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">工具支持:</span>{' '}
                      <span className={currentConfig.supports_tools ? 'text-green-600' : 'text-gray-400'}>
                        {currentConfig.supports_tools ? '是' : '否'}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Provider Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  LLM 提供商
                </label>
                <div className="relative">
                  <select
                    value={selectedProvider}
                    onChange={(e) => handleProviderChange(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg appearance-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 bg-white"
                  >
                    <option value="">选择提供商...</option>
                    {providers.map((provider) => (
                      <option key={provider.id} value={provider.id}>
                        {provider.name} {provider.requires_api_key ? '(需要 API Key)' : '(本地)'}
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                </div>
              </div>

              {/* Model Selection */}
              {selectedProvider && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    模型
                  </label>
                  <div className="relative">
                    <select
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg appearance-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 bg-white"
                    >
                      {currentProvider?.models.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </select>
                    <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                  </div>
                </div>
              )}

              {/* API Key (if required) */}
              {requiresApiKey && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    API Key
                    <span className="text-red-500 ml-1">*</span>
                  </label>
                  <div className="relative">
                    <input
                      type={showApiKey ? 'text' : 'password'}
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder="输入 API Key..."
                      className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                    <button
                      type="button"
                      onClick={() => setShowApiKey(!showApiKey)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                    >
                      {showApiKey ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                  </div>
                  <p className="mt-1 text-xs text-gray-500">
                    API Key 仅在内存中使用，不会持久存储
                  </p>
                </div>
              )}

              {/* Custom Base URL (optional) */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  自定义 Base URL
                  <span className="text-gray-400 ml-1">(可选)</span>
                </label>
                <input
                  type="text"
                  value={customBaseUrl}
                  onChange={(e) => setCustomBaseUrl(e.target.value)}
                  placeholder={currentProvider?.base_url || '使用默认 URL'}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
              </div>

              {/* Temperature */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  温度: {temperature}
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>精确 (0)</span>
                  <span>平衡 (0.7)</span>
                  <span>创意 (2)</span>
                </div>
              </div>

              {/* Max Tokens */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  最大 Tokens
                </label>
                <input
                  type="number"
                  min="1"
                  max="128000"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value) || 2048)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
              </div>

              {/* Test Result */}
              {testResult && (
                <div className={`p-4 rounded-lg ${testResult.success ? 'bg-green-50' : 'bg-red-50'}`}>
                  <div className="flex items-start gap-2">
                    {testResult.success ? (
                      <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                    ) : (
                      <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                    )}
                    <p className={`text-sm ${testResult.success ? 'text-green-700' : 'text-red-700'}`}>
                      {testResult.message}
                    </p>
                  </div>
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="p-4 bg-red-50 rounded-lg">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-5 h-5 text-red-500" />
                    <p className="text-sm text-red-700">{error}</p>
                  </div>
                </div>
              )}

              {/* Success Message */}
              {success && (
                <div className="p-4 bg-green-50 rounded-lg">
                  <div className="flex items-center gap-2">
                    <Check className="w-5 h-5 text-green-500" />
                    <p className="text-sm text-green-700">{success}</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-200 bg-gray-50">
          <button
            onClick={handleTest}
            disabled={!selectedProvider || !selectedModel || isTesting || isLoading}
            className="flex items-center gap-2 px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isTesting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            测试连接
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            取消
          </button>
          <button
            onClick={handleSave}
            disabled={!selectedProvider || !selectedModel || isSaving || isLoading}
            className="flex items-center gap-2 px-4 py-2 text-white bg-primary-500 rounded-lg hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isSaving ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Check className="w-4 h-4" />
            )}
            保存配置
          </button>
        </div>
      </div>
    </div>
  );
};

export default ModelConfigPanel;
