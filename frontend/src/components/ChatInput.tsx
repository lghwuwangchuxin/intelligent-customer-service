import React, { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Settings } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string) => void;
  onFileUpload?: (file: File) => void;
  disabled?: boolean;
  placeholder?: string;
  useRag: boolean;
  onToggleRag: (value: boolean) => void;
}

const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  onFileUpload,
  disabled = false,
  placeholder = '请输入您的问题...',
  useRag,
  onToggleRag,
}) => {
  const [message, setMessage] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        120
      )}px`;
    }
  }, [message]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSend(message.trim());
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && onFileUpload) {
      onFileUpload(file);
    }
    e.target.value = '';
  };

  return (
    <div className="border-t border-gray-200 bg-white p-4">
      {/* Settings panel */}
      {showSettings && (
        <div className="mb-3 p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">使用知识库问答</span>
            <button
              onClick={() => onToggleRag(!useRag)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                useRag ? 'bg-primary-500' : 'bg-gray-300'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  useRag ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-1">
            {useRag
              ? '已启用: 将根据知识库内容回答问题'
              : '已禁用: 使用通用对话模式'}
          </p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="flex items-end gap-2">
        {/* File upload button */}
        {onFileUpload && (
          <>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".pdf,.docx,.doc,.xlsx,.xls,.txt,.md"
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex-shrink-0 p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              title="上传文档"
            >
              <Paperclip className="w-5 h-5" />
            </button>
          </>
        )}

        {/* Settings button */}
        <button
          type="button"
          onClick={() => setShowSettings(!showSettings)}
          className={`flex-shrink-0 p-2 rounded-lg transition-colors ${
            showSettings
              ? 'text-primary-500 bg-primary-50'
              : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
          }`}
          title="设置"
        >
          <Settings className="w-5 h-5" />
        </button>

        {/* Input area */}
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            className="w-full px-4 py-2 pr-12 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
          />
        </div>

        {/* Send button */}
        <button
          type="submit"
          disabled={!message.trim() || disabled}
          className="flex-shrink-0 p-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          <Send className="w-5 h-5" />
        </button>
      </form>
    </div>
  );
};

export default ChatInput;
