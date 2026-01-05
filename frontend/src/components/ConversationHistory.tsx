import React, { useState, useEffect, useCallback } from 'react';
import {
  MessageSquare,
  Plus,
  Trash2,
  Edit2,
  Check,
  X,
  Clock,
  ChevronRight,
  Download,
  RefreshCw,
} from 'lucide-react';
import { conversationApi, ConversationSummary } from '../services/api';

interface ConversationHistoryProps {
  currentConversationId?: string;
  onSelectConversation: (conversationId: string) => void;
  onNewConversation: () => void;
  onDeleteConversation?: (conversationId: string) => void;
}

export const ConversationHistory: React.FC<ConversationHistoryProps> = ({
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
}) => {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');

  const loadConversations = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await conversationApi.list({ limit: 50 });
      // Sort by updated_at in descending order
      const sorted = response.conversations.sort((a, b) =>
        new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
      );
      setConversations(sorted);
    } catch (err) {
      setError('加载对话历史失败');
      console.error('Failed to load conversations:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  const handleDelete = async (e: React.MouseEvent, conversationId: string) => {
    e.stopPropagation();
    if (!confirm('确定要删除这个对话吗？')) return;

    try {
      await conversationApi.delete(conversationId);
      setConversations((prev) => prev.filter((c) => c.conversation_id !== conversationId));
      onDeleteConversation?.(conversationId);
    } catch (err) {
      console.error('Failed to delete conversation:', err);
    }
  };

  const handleStartEdit = (e: React.MouseEvent, conv: ConversationSummary) => {
    e.stopPropagation();
    setEditingId(conv.conversation_id);
    setEditTitle(conv.title || getDefaultTitle(conv));
  };

  const handleSaveEdit = async (e: React.MouseEvent, conversationId: string) => {
    e.stopPropagation();
    try {
      await conversationApi.update(conversationId, { title: editTitle });
      setConversations((prev) =>
        prev.map((c) =>
          c.conversation_id === conversationId ? { ...c, title: editTitle } : c
        )
      );
    } catch (err) {
      console.error('Failed to update conversation:', err);
    } finally {
      setEditingId(null);
    }
  };

  const handleCancelEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(null);
    setEditTitle('');
  };

  const handleExport = async (e: React.MouseEvent, conversationId: string) => {
    e.stopPropagation();
    try {
      const result = await conversationApi.export(conversationId);
      const blob = new Blob([JSON.stringify(result.data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `conversation-${conversationId}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to export conversation:', err);
    }
  };

  const getDefaultTitle = (conv: ConversationSummary) => {
    if (conv.last_message?.content) {
      return conv.last_message.content.substring(0, 30) + (conv.last_message.content.length > 30 ? '...' : '');
    }
    return '新对话';
  };

  const formatTime = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) {
      return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
    } else if (days === 1) {
      return '昨天';
    } else if (days < 7) {
      return `${days}天前`;
    } else {
      return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
    }
  };

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="p-3 border-b bg-white">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-medium text-gray-900 flex items-center gap-2">
            <MessageSquare size={18} />
            对话历史
          </h3>
          <button
            onClick={loadConversations}
            className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded"
            title="刷新"
          >
            <RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
          </button>
        </div>
        <button
          onClick={onNewConversation}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
        >
          <Plus size={18} />
          新建对话
        </button>
      </div>

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto p-2">
        {error && (
          <div className="p-3 mb-2 bg-red-50 text-red-600 rounded-lg text-sm">
            {error}
          </div>
        )}

        {isLoading && conversations.length === 0 ? (
          <div className="flex items-center justify-center py-8 text-gray-500">
            <RefreshCw size={20} className="animate-spin mr-2" />
            加载中...
          </div>
        ) : conversations.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <MessageSquare size={40} className="mx-auto mb-2 opacity-50" />
            <p>暂无对话历史</p>
            <p className="text-sm mt-1">点击上方按钮开始新对话</p>
          </div>
        ) : (
          <div className="space-y-1">
            {conversations.map((conv) => (
              <div
                key={conv.conversation_id}
                onClick={() => onSelectConversation(conv.conversation_id)}
                className={`group p-3 rounded-lg cursor-pointer transition-all ${
                  currentConversationId === conv.conversation_id
                    ? 'bg-blue-100 border border-blue-200'
                    : 'bg-white hover:bg-gray-100 border border-transparent'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  {editingId === conv.conversation_id ? (
                    <div className="flex-1 flex items-center gap-1">
                      <input
                        type="text"
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                        className="flex-1 px-2 py-1 text-sm border rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                        onClick={(e) => e.stopPropagation()}
                        autoFocus
                      />
                      <button
                        onClick={(e) => handleSaveEdit(e, conv.conversation_id)}
                        className="p-1 text-green-600 hover:bg-green-50 rounded"
                      >
                        <Check size={14} />
                      </button>
                      <button
                        onClick={handleCancelEdit}
                        className="p-1 text-gray-500 hover:bg-gray-100 rounded"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  ) : (
                    <>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-gray-900 truncate">
                          {conv.title || getDefaultTitle(conv)}
                        </p>
                        <div className="flex items-center gap-2 mt-1 text-xs text-gray-500">
                          <span className="flex items-center gap-1">
                            <Clock size={12} />
                            {formatTime(conv.updated_at)}
                          </span>
                          <span>{conv.message_count} 条消息</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={(e) => handleStartEdit(e, conv)}
                          className="p-1 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded"
                          title="重命名"
                        >
                          <Edit2 size={14} />
                        </button>
                        <button
                          onClick={(e) => handleExport(e, conv.conversation_id)}
                          className="p-1 text-gray-500 hover:text-green-600 hover:bg-green-50 rounded"
                          title="导出"
                        >
                          <Download size={14} />
                        </button>
                        <button
                          onClick={(e) => handleDelete(e, conv.conversation_id)}
                          className="p-1 text-gray-500 hover:text-red-600 hover:bg-red-50 rounded"
                          title="删除"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </>
                  )}
                </div>
                {conv.has_summary && (
                  <div className="mt-2 flex items-center gap-1 text-xs text-blue-600">
                    <ChevronRight size={12} />
                    已生成摘要
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ConversationHistory;
