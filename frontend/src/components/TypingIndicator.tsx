import React from 'react';
import { Bot } from 'lucide-react';

const TypingIndicator: React.FC = () => {
  return (
    <div className="flex gap-3 p-4 bg-gray-50 message-fade-in">
      {/* Avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-green-500 flex items-center justify-center">
        <Bot className="w-5 h-5 text-white" />
      </div>

      {/* Typing dots */}
      <div className="flex items-center gap-1 py-2">
        <div className="w-2 h-2 bg-gray-400 rounded-full typing-dot" />
        <div className="w-2 h-2 bg-gray-400 rounded-full typing-dot" />
        <div className="w-2 h-2 bg-gray-400 rounded-full typing-dot" />
      </div>
    </div>
  );
};

export default TypingIndicator;
