import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Upload, FileText, Search, X, Check, AlertCircle, Database, RefreshCw, BarChart3 } from 'lucide-react';
import { ragApi, RAGStats, DocumentInfo } from '../services/api';

interface KnowledgePanelProps {
  isOpen: boolean;
  onClose: () => void;
}

const KnowledgePanel: React.FC<KnowledgePanelProps> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState<'upload' | 'text' | 'search' | 'stats'>('upload');
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [textContent, setTextContent] = useState('');
  const [textTitle, setTextTitle] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<DocumentInfo[]>([]);
  const [stats, setStats] = useState<RAGStats | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadStats = useCallback(async () => {
    try {
      const data = await ragApi.getStats();
      setStats(data);
    } catch (err) {
      console.error('Failed to load RAG stats:', err);
    }
  }, []);

  useEffect(() => {
    if (isOpen && activeTab === 'stats') {
      loadStats();
    }
  }, [isOpen, activeTab, loadStats]);

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    setMessage(null);
    try {
      const result = await ragApi.uploadFile(file);
      if (result.success) {
        setMessage({ type: 'success', text: `成功上传 ${file.name}，已创建 ${result.chunks_created || 0} 个文档片段` });
        loadStats();
      } else {
        setMessage({ type: 'error', text: result.error || '上传失败' });
      }
    } catch (error) {
      console.error('Upload error:', error);
      setMessage({ type: 'error', text: '上传失败，请检查服务是否正常运行' });
    } finally {
      setUploading(false);
    }
  };

  const handleAddText = async () => {
    if (!textContent.trim()) return;

    setUploading(true);
    setMessage(null);
    try {
      const result = await ragApi.indexDocument({
        content: textContent,
        source: textTitle || undefined,
      });
      if (result.success) {
        setMessage({ type: 'success', text: `成功添加文本，已创建 ${result.chunks_created || 0} 个文档片段` });
        setTextContent('');
        setTextTitle('');
        loadStats();
      } else {
        setMessage({ type: 'error', text: result.error || '添加失败' });
      }
    } catch (error) {
      console.error('Add text error:', error);
      setMessage({ type: 'error', text: '添加失败，请检查服务是否正常运行' });
    } finally {
      setUploading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    setMessage(null);
    try {
      const response = await ragApi.retrieve({
        query: searchQuery,
        top_k: 5,
        config: {
          enable_rerank: true,
        },
      });
      setSearchResults(response.documents);
      if (response.documents.length === 0) {
        setMessage({ type: 'error', text: '未找到相关结果' });
      }
    } catch (error) {
      console.error('Search error:', error);
      setMessage({ type: 'error', text: '搜索失败，请检查服务是否正常运行' });
    } finally {
      setIsSearching(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <h2 className="text-lg font-semibold text-gray-900">知识库管理</h2>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-gray-600 rounded"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b">
          <button
            onClick={() => setActiveTab('upload')}
            className={`flex-1 py-3 text-sm font-medium ${
              activeTab === 'upload'
                ? 'text-primary-600 border-b-2 border-primary-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <Upload className="w-4 h-4 inline mr-2" />
            上传文档
          </button>
          <button
            onClick={() => setActiveTab('text')}
            className={`flex-1 py-3 text-sm font-medium ${
              activeTab === 'text'
                ? 'text-primary-600 border-b-2 border-primary-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <FileText className="w-4 h-4 inline mr-2" />
            添加文本
          </button>
          <button
            onClick={() => setActiveTab('search')}
            className={`flex-1 py-3 text-sm font-medium ${
              activeTab === 'search'
                ? 'text-primary-600 border-b-2 border-primary-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <Search className="w-4 h-4 inline mr-2" />
            搜索测试
          </button>
          <button
            onClick={() => setActiveTab('stats')}
            className={`flex-1 py-3 text-sm font-medium ${
              activeTab === 'stats'
                ? 'text-primary-600 border-b-2 border-primary-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <BarChart3 className="w-4 h-4 inline mr-2" />
            统计
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {/* Message */}
          {message && (
            <div
              className={`mb-4 p-3 rounded-lg flex items-center gap-2 ${
                message.type === 'success'
                  ? 'bg-green-50 text-green-700'
                  : 'bg-red-50 text-red-700'
              }`}
            >
              {message.type === 'success' ? (
                <Check className="w-5 h-5" />
              ) : (
                <AlertCircle className="w-5 h-5" />
              )}
              {message.text}
            </div>
          )}

          {/* Upload Tab */}
          {activeTab === 'upload' && (
            <div>
              <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFileUpload(file);
                  e.target.value = '';
                }}
                accept=".pdf,.docx,.doc,.xlsx,.xls,.txt,.md"
                className="hidden"
              />
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-primary-500 hover:bg-primary-50 transition-colors"
              >
                <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 mb-2">点击或拖拽文件到此处上传</p>
                <p className="text-sm text-gray-400">
                  支持 PDF, Word, Excel, TXT, Markdown 格式
                </p>
              </div>

              <div className="mt-6 p-3 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-600 mb-1">支持格式:</p>
                <div className="flex flex-wrap gap-2">
                  <span className="px-2 py-0.5 bg-white rounded text-xs text-gray-500">PDF</span>
                  <span className="px-2 py-0.5 bg-white rounded text-xs text-gray-500">Word (DOCX)</span>
                  <span className="px-2 py-0.5 bg-white rounded text-xs text-gray-500">Excel (XLSX)</span>
                  <span className="px-2 py-0.5 bg-white rounded text-xs text-gray-500">TXT</span>
                  <span className="px-2 py-0.5 bg-white rounded text-xs text-gray-500">Markdown</span>
                </div>
              </div>
            </div>
          )}

          {/* Text Tab */}
          {activeTab === 'text' && (
            <div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  标题（可选）
                </label>
                <input
                  type="text"
                  value={textTitle}
                  onChange={(e) => setTextTitle(e.target.value)}
                  placeholder="文档标题"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  内容
                </label>
                <textarea
                  value={textContent}
                  onChange={(e) => setTextContent(e.target.value)}
                  placeholder="输入要添加到知识库的文本内容..."
                  rows={8}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>
              <button
                onClick={handleAddText}
                disabled={!textContent.trim() || uploading}
                className="w-full py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                {uploading ? '处理中...' : '添加到知识库'}
              </button>
            </div>
          )}

          {/* Search Tab */}
          {activeTab === 'search' && (
            <div>
              <div className="flex gap-2 mb-4">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="输入搜索关键词..."
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <button
                  onClick={handleSearch}
                  disabled={isSearching}
                  className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isSearching ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    '搜索'
                  )}
                </button>
              </div>

              {searchResults.length > 0 && (
                <div className="space-y-3">
                  {searchResults.map((result, index) => (
                    <div
                      key={index}
                      className="p-3 bg-gray-50 rounded-lg border border-gray-200"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-gray-500">
                          来源: {result.source || result.metadata?.source || '未知'}
                        </span>
                        <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-600 rounded">
                          相关度: {(result.score * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="text-sm text-gray-700 line-clamp-3">{result.content}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Stats Tab */}
          {activeTab === 'stats' && (
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-gray-900 flex items-center gap-2">
                  <Database className="w-4 h-4 text-primary-500" />
                  知识库统计
                </h3>
                <button
                  onClick={loadStats}
                  className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>

              {stats ? (
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <p className="text-2xl font-bold text-blue-600">{stats.total_documents}</p>
                    <p className="text-sm text-gray-600">文档总数</p>
                  </div>
                  <div className="p-4 bg-green-50 rounded-lg">
                    <p className="text-2xl font-bold text-green-600">{stats.total_chunks}</p>
                    <p className="text-sm text-gray-600">文档片段</p>
                  </div>
                  <div className="p-4 bg-purple-50 rounded-lg col-span-2">
                    <p className="text-2xl font-bold text-purple-600">
                      {stats.index_size_bytes ? (stats.index_size_bytes / 1024 / 1024).toFixed(2) : 0} MB
                    </p>
                    <p className="text-sm text-gray-600">索引大小</p>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <RefreshCw className="w-8 h-8 mx-auto mb-2 animate-spin" />
                  <p>加载统计数据...</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default KnowledgePanel;
