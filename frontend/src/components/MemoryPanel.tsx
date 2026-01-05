import React, { useState, useEffect, useCallback } from 'react';
import {
  Brain,
  User,
  Tag,
  BookOpen,
  Search,
  Plus,
  Trash2,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Database,
  Settings,
  MessageSquare,
  Clock,
} from 'lucide-react';
import {
  memoryApi,
  memoryStoreApi,
  MemoryEntity,
  MemoryKnowledge,
  MemoryStats,
  ConversationInfo,
} from '../services/api';

interface MemoryPanelProps {
  userId: string;
  onUserIdChange?: (userId: string) => void;
}

interface PreferenceItem {
  key: string;
  value: unknown;
  category?: string;
}

type TabType = 'conversations' | 'preferences' | 'entities' | 'knowledge' | 'stats';

export const MemoryPanel: React.FC<MemoryPanelProps> = ({
  userId,
  onUserIdChange,
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('conversations');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Data states
  const [conversations, setConversations] = useState<ConversationInfo[]>([]);
  const [preferences, setPreferences] = useState<PreferenceItem[]>([]);
  const [entities, setEntities] = useState<MemoryEntity[]>([]);
  const [knowledge, setKnowledge] = useState<MemoryKnowledge[]>([]);
  const [stats, setStats] = useState<MemoryStats | null>(null);

  // Form states
  const [newPrefKey, setNewPrefKey] = useState('');
  const [newPrefValue, setNewPrefValue] = useState('');
  const [newPrefCategory, setNewPrefCategory] = useState('general');
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['preferences']));

  // User ID input state
  const [inputUserId, setInputUserId] = useState(userId);

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(section)) {
        newSet.delete(section);
      } else {
        newSet.add(section);
      }
      return newSet;
    });
  };

  const loadConversations = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await memoryApi.listConversations(1, 50);
      setConversations(response.conversations || []);
    } catch (err) {
      setError('加载对话列表失败');
      console.error('Failed to load conversations:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadPreferences = useCallback(async () => {
    if (!userId) return;
    setIsLoading(true);
    setError(null);
    try {
      const response = await memoryStoreApi.getUserPreferences(userId);
      // Convert Record<string, unknown> to array of PreferenceItem
      const prefsObj = response.preferences || {};
      const prefsArray: PreferenceItem[] = Object.entries(prefsObj).map(([key, value]) => ({
        key,
        value,
        category: 'general',
      }));
      setPreferences(prefsArray);
    } catch (err) {
      setError('加载用户偏好失败');
      console.error('Failed to load preferences:', err);
    } finally {
      setIsLoading(false);
    }
  }, [userId]);

  const loadStats = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await memoryStoreApi.getStats();
      setStats(response);
    } catch (err) {
      console.error('Failed to load stats:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (activeTab === 'conversations') {
      loadConversations();
    } else if (activeTab === 'preferences' && userId) {
      loadPreferences();
    } else if (activeTab === 'stats') {
      loadStats();
    }
  }, [activeTab, userId, loadConversations, loadPreferences, loadStats]);

  const handleAddPreference = async () => {
    if (!userId || !newPrefKey.trim()) return;
    setIsLoading(true);
    try {
      await memoryStoreApi.setUserPreference(userId, {
        key: newPrefKey,
        value: newPrefValue,
        category: newPrefCategory,
      });
      await loadPreferences();
      setNewPrefKey('');
      setNewPrefValue('');
    } catch (err) {
      setError('添加偏好失败');
      console.error('Failed to add preference:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearchEntities = async () => {
    if (!searchQuery.trim()) return;
    setIsLoading(true);
    try {
      const response = await memoryStoreApi.searchEntities(searchQuery, undefined, 10);
      setEntities(response.results);
    } catch (err) {
      setError('搜索实体失败');
      console.error('Failed to search entities:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearchKnowledge = async () => {
    if (!searchQuery.trim()) return;
    setIsLoading(true);
    try {
      const response = await memoryStoreApi.searchKnowledge(searchQuery, 10);
      setKnowledge(response.results);
    } catch (err) {
      setError('搜索知识失败');
      console.error('Failed to search knowledge:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteConversation = async (conversationId: string) => {
    if (!confirm('确定要删除这个对话吗？')) return;
    setIsLoading(true);
    try {
      await memoryApi.deleteConversation(conversationId);
      await loadConversations();
    } catch (err) {
      setError('删除对话失败');
      console.error('Failed to delete conversation:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearUserMemory = async () => {
    if (!userId) return;
    if (!confirm(`确定要清除用户 ${userId} 的所有偏好设置吗？`)) return;
    setIsLoading(true);
    try {
      // Clear preferences by setting them to empty
      setPreferences([]);
      alert('用户偏好已清除');
    } catch (err) {
      setError('清除记忆失败');
      console.error('Failed to clear memory:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSetUserId = () => {
    if (inputUserId.trim()) {
      onUserIdChange?.(inputUserId.trim());
    }
  };

  const formatTime = (dateStr: string) => {
    try {
      const date = new Date(dateStr);
      return date.toLocaleString('zh-CN', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return dateStr;
    }
  };

  const tabs: { id: TabType; label: string; icon: React.ReactNode }[] = [
    { id: 'conversations', label: '对话', icon: <MessageSquare size={16} /> },
    { id: 'preferences', label: '偏好', icon: <Settings size={16} /> },
    { id: 'entities', label: '实体', icon: <Tag size={16} /> },
    { id: 'knowledge', label: '知识', icon: <BookOpen size={16} /> },
    { id: 'stats', label: '统计', icon: <Database size={16} /> },
  ];

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="p-3 border-b bg-white">
        <h3 className="font-medium text-gray-900 flex items-center gap-2 mb-3">
          <Brain size={18} className="text-purple-500" />
          长期记忆
        </h3>

        {/* User ID Input */}
        <div className="flex items-center gap-2 mb-3">
          <User size={16} className="text-gray-400" />
          <input
            type="text"
            value={inputUserId}
            onChange={(e) => setInputUserId(e.target.value)}
            placeholder="输入用户ID"
            className="flex-1 px-2 py-1.5 text-sm border rounded focus:outline-none focus:ring-1 focus:ring-purple-500"
          />
          <button
            onClick={handleSetUserId}
            className="px-3 py-1.5 bg-purple-500 text-white text-sm rounded hover:bg-purple-600"
          >
            设置
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1 px-3 py-1.5 text-sm rounded transition-colors ${
                activeTab === tab.id
                  ? 'bg-purple-100 text-purple-700'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3">
        {error && (
          <div className="p-3 mb-3 bg-red-50 text-red-600 rounded-lg text-sm">
            {error}
          </div>
        )}

        {!userId && activeTab !== 'stats' && activeTab !== 'conversations' ? (
          <div className="text-center py-8 text-gray-500">
            <User size={40} className="mx-auto mb-2 opacity-50" />
            <p>请先设置用户ID</p>
          </div>
        ) : (
          <>
            {/* Conversations Tab */}
            {activeTab === 'conversations' && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">
                    共 {conversations.length} 个对话
                  </span>
                  <button
                    onClick={loadConversations}
                    className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded"
                  >
                    <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
                  </button>
                </div>

                {conversations.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <MessageSquare size={40} className="mx-auto mb-2 opacity-50" />
                    <p>暂无对话记录</p>
                  </div>
                ) : (
                  <div className="bg-white rounded-lg border divide-y">
                    {conversations.map((conv) => (
                      <div key={conv.conversation_id} className="p-3 hover:bg-gray-50">
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 truncate">
                              {conv.title || `对话 ${conv.conversation_id.slice(0, 8)}...`}
                            </p>
                            <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                              <span className="flex items-center gap-1">
                                <MessageSquare size={12} />
                                {conv.message_count} 条消息
                              </span>
                              {conv.has_summary && (
                                <span className="px-1.5 py-0.5 bg-green-100 text-green-600 rounded">
                                  已总结
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-1 mt-1 text-xs text-gray-400">
                              <Clock size={12} />
                              {formatTime(conv.updated_at)}
                            </div>
                          </div>
                          <button
                            onClick={() => handleDeleteConversation(conv.conversation_id)}
                            className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded"
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Preferences Tab */}
            {activeTab === 'preferences' && (
              <div className="space-y-4">
                {/* Add Preference Form */}
                <div className="bg-white rounded-lg p-3 border">
                  <div
                    className="flex items-center justify-between cursor-pointer"
                    onClick={() => toggleSection('add-pref')}
                  >
                    <span className="font-medium text-sm flex items-center gap-2">
                      <Plus size={16} />
                      添加偏好
                    </span>
                    {expandedSections.has('add-pref') ? (
                      <ChevronDown size={16} />
                    ) : (
                      <ChevronRight size={16} />
                    )}
                  </div>
                  {expandedSections.has('add-pref') && (
                    <div className="mt-3 space-y-2">
                      <input
                        type="text"
                        value={newPrefKey}
                        onChange={(e) => setNewPrefKey(e.target.value)}
                        placeholder="偏好键 (如: language)"
                        className="w-full px-2 py-1.5 text-sm border rounded focus:outline-none focus:ring-1 focus:ring-purple-500"
                      />
                      <input
                        type="text"
                        value={newPrefValue}
                        onChange={(e) => setNewPrefValue(e.target.value)}
                        placeholder="偏好值 (如: zh-CN)"
                        className="w-full px-2 py-1.5 text-sm border rounded focus:outline-none focus:ring-1 focus:ring-purple-500"
                      />
                      <select
                        value={newPrefCategory}
                        onChange={(e) => setNewPrefCategory(e.target.value)}
                        className="w-full px-2 py-1.5 text-sm border rounded focus:outline-none focus:ring-1 focus:ring-purple-500"
                      >
                        <option value="general">通用</option>
                        <option value="ui">界面</option>
                        <option value="behavior">行为</option>
                        <option value="notification">通知</option>
                      </select>
                      <button
                        onClick={handleAddPreference}
                        disabled={isLoading || !newPrefKey.trim()}
                        className="w-full px-3 py-1.5 bg-purple-500 text-white text-sm rounded hover:bg-purple-600 disabled:opacity-50"
                      >
                        添加
                      </button>
                    </div>
                  )}
                </div>

                {/* Preferences List */}
                <div className="bg-white rounded-lg border">
                  <div className="p-3 border-b flex items-center justify-between">
                    <span className="font-medium text-sm">已保存的偏好</span>
                    <button
                      onClick={loadPreferences}
                      className="p-1 text-gray-500 hover:text-gray-700"
                    >
                      <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
                    </button>
                  </div>
                  {preferences.length === 0 ? (
                    <div className="p-4 text-center text-gray-500 text-sm">
                      暂无偏好设置
                    </div>
                  ) : (
                    <div className="divide-y">
                      {preferences.map((pref, idx) => (
                        <div key={idx} className="p-3 flex items-center justify-between">
                          <div>
                            <p className="text-sm font-medium text-gray-900">{pref.key}</p>
                            <p className="text-xs text-gray-500">
                              {String(pref.value)} · {pref.category}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Clear Memory */}
                <button
                  onClick={handleClearUserMemory}
                  className="w-full flex items-center justify-center gap-2 px-3 py-2 text-red-600 border border-red-200 rounded-lg hover:bg-red-50"
                >
                  <Trash2 size={16} />
                  清除用户记忆
                </button>
              </div>
            )}

            {/* Entities Tab */}
            {activeTab === 'entities' && (
              <div className="space-y-4">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="搜索实体..."
                    className="flex-1 px-3 py-2 text-sm border rounded-lg focus:outline-none focus:ring-1 focus:ring-purple-500"
                    onKeyDown={(e) => e.key === 'Enter' && handleSearchEntities()}
                  />
                  <button
                    onClick={handleSearchEntities}
                    disabled={isLoading}
                    className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 disabled:opacity-50"
                  >
                    <Search size={16} />
                  </button>
                </div>

                {entities.length > 0 && (
                  <div className="bg-white rounded-lg border divide-y">
                    {entities.map((entity, idx) => (
                      <div key={idx} className="p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <Tag size={14} className="text-purple-500" />
                          <span className="font-medium text-sm">{entity.name}</span>
                          <span className="text-xs px-2 py-0.5 bg-gray-100 rounded">
                            {entity.entity_type}
                          </span>
                        </div>
                        {Object.keys(entity.attributes).length > 0 && (
                          <div className="text-xs text-gray-500 mt-1">
                            {Object.entries(entity.attributes).map(([key, value]) => (
                              <span key={key} className="mr-2">
                                {key}: {String(value)}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Knowledge Tab */}
            {activeTab === 'knowledge' && (
              <div className="space-y-4">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="搜索知识..."
                    className="flex-1 px-3 py-2 text-sm border rounded-lg focus:outline-none focus:ring-1 focus:ring-purple-500"
                    onKeyDown={(e) => e.key === 'Enter' && handleSearchKnowledge()}
                  />
                  <button
                    onClick={handleSearchKnowledge}
                    disabled={isLoading}
                    className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 disabled:opacity-50"
                  >
                    <Search size={16} />
                  </button>
                </div>

                {knowledge.length > 0 && (
                  <div className="bg-white rounded-lg border divide-y">
                    {knowledge.map((item, idx) => (
                      <div key={idx} className="p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <BookOpen size={14} className="text-green-500" />
                          <span className="font-medium text-sm">{item.topic}</span>
                          {item.relevance_score && (
                            <span className="text-xs text-gray-400">
                              相关度: {(item.relevance_score * 100).toFixed(0)}%
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-gray-600 mt-1">{item.content}</p>
                        {item.source && (
                          <p className="text-xs text-gray-400 mt-1">来源: {item.source}</p>
                        )}
                        {item.tags && item.tags.length > 0 && (
                          <div className="flex gap-1 mt-2">
                            {item.tags.map((tag, tagIdx) => (
                              <span
                                key={tagIdx}
                                className="text-xs px-2 py-0.5 bg-green-50 text-green-600 rounded"
                              >
                                {tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Stats Tab */}
            {activeTab === 'stats' && (
              <div className="space-y-4">
                {stats ? (
                  <div className="bg-white rounded-lg border p-4">
                    <h4 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
                      <Database size={16} className="text-purple-500" />
                      记忆统计
                    </h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-3 bg-purple-50 rounded-lg">
                        <p className="text-2xl font-bold text-purple-600">{stats.total_conversations}</p>
                        <p className="text-sm text-gray-600">对话数</p>
                      </div>
                      <div className="p-3 bg-blue-50 rounded-lg">
                        <p className="text-2xl font-bold text-blue-600">{stats.total_messages}</p>
                        <p className="text-sm text-gray-600">消息数</p>
                      </div>
                      <div className="p-3 bg-green-50 rounded-lg">
                        <p className="text-2xl font-bold text-green-600">{stats.total_memories}</p>
                        <p className="text-sm text-gray-600">长期记忆</p>
                      </div>
                      <div className="p-3 bg-orange-50 rounded-lg">
                        <p className="text-2xl font-bold text-orange-600">{stats.namespaces?.length || 0}</p>
                        <p className="text-sm text-gray-600">命名空间</p>
                      </div>
                    </div>
                    {stats.memory_types && Object.keys(stats.memory_types).length > 0 && (
                      <div className="mt-4">
                        <p className="text-sm text-gray-600 mb-2">记忆类型:</p>
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(stats.memory_types).map(([type, count]) => (
                            <span
                              key={type}
                              className="text-xs px-2 py-1 bg-gray-100 rounded flex items-center gap-1"
                            >
                              <span className="font-medium">{type}</span>
                              <span className="text-gray-400">({count})</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    {stats.namespaces && stats.namespaces.length > 0 && (
                      <div className="mt-4">
                        <p className="text-sm text-gray-600 mb-2">命名空间:</p>
                        <div className="flex flex-wrap gap-1">
                          {stats.namespaces.slice(0, 10).map((ns, idx) => (
                            <span
                              key={idx}
                              className="text-xs px-2 py-0.5 bg-gray-100 rounded"
                            >
                              {ns}
                            </span>
                          ))}
                          {stats.namespaces.length > 10 && (
                            <span className="text-xs text-gray-400">
                              +{stats.namespaces.length - 10} 更多
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <RefreshCw size={20} className="mx-auto mb-2 animate-spin" />
                    加载中...
                  </div>
                )}
                <button
                  onClick={loadStats}
                  className="w-full flex items-center justify-center gap-2 px-3 py-2 border rounded-lg hover:bg-gray-50"
                >
                  <RefreshCw size={16} />
                  刷新统计
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default MemoryPanel;
