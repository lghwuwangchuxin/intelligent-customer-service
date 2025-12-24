import React, { useState, useRef } from 'react';
import { Upload, FileText, Trash2, Search, X, Check, AlertCircle } from 'lucide-react';
import { knowledgeApi, KnowledgeAddRequest } from '../services/api';

interface KnowledgePanelProps {
  isOpen: boolean;
  onClose: () => void;
}

const KnowledgePanel: React.FC<KnowledgePanelProps> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState<'upload' | 'text' | 'search'>('upload');
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [textContent, setTextContent] = useState('');
  const [textTitle, setTextTitle] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Array<{ content: string; source: string }>>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    setMessage(null);
    try {
      const result = await knowledgeApi.uploadDocument(file);
      if (result.success) {
        setMessage({ type: 'success', text: `成功上传 ${file.name}，已索引 ${result.num_documents} 个文档片段` });
      } else {
        setMessage({ type: 'error', text: result.message });
      }
    } catch (error) {
      setMessage({ type: 'error', text: '上传失败，请重试' });
    } finally {
      setUploading(false);
    }
  };

  const handleAddText = async () => {
    if (!textContent.trim()) return;

    setUploading(true);
    setMessage(null);
    try {
      const request: KnowledgeAddRequest = {
        text: textContent,
        title: textTitle || undefined,
      };
      const result = await knowledgeApi.addText(request);
      if (result.success) {
        setMessage({ type: 'success', text: `成功添加文本，已索引 ${result.num_documents} 个文档片段` });
        setTextContent('');
        setTextTitle('');
      } else {
        setMessage({ type: 'error', text: result.message });
      }
    } catch (error) {
      setMessage({ type: 'error', text: '添加失败，请重试' });
    } finally {
      setUploading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    try {
      const results = await knowledgeApi.search(searchQuery, 5);
      setSearchResults(results);
    } catch (error) {
      setMessage({ type: 'error', text: '搜索失败' });
    }
  };

  const handleClear = async () => {
    if (!confirm('确定要清空知识库吗？此操作不可恢复。')) return;

    try {
      await knowledgeApi.clear();
      setMessage({ type: 'success', text: '知识库已清空' });
      setSearchResults([]);
    } catch (error) {
      setMessage({ type: 'error', text: '清空失败' });
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

              <div className="mt-6 flex justify-end">
                <button
                  onClick={handleClear}
                  className="flex items-center gap-2 px-4 py-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                  清空知识库
                </button>
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
                  className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
                >
                  搜索
                </button>
              </div>

              {searchResults.length > 0 && (
                <div className="space-y-3">
                  {searchResults.map((result, index) => (
                    <div
                      key={index}
                      className="p-3 bg-gray-50 rounded-lg border border-gray-200"
                    >
                      <div className="text-xs text-gray-500 mb-1">
                        来源: {result.source}
                      </div>
                      <div className="text-sm text-gray-700">{result.content}</div>
                    </div>
                  ))}
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
