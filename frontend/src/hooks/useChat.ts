import { useState, useCallback, useRef } from 'react';
import { chatApi, ChatMessage as ApiChatMessage } from '../services/api';
import { Message } from '../components/ChatMessage';

export interface UseChatOptions {
  useRag?: boolean;
}

export function useChat(options: UseChatOptions = {}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useRag, setUseRag] = useState(options.useRag ?? true);
  const abortControllerRef = useRef<AbortController | null>(null);

  const generateId = () => `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  const sendMessage = useCallback(
    async (content: string, stream: boolean = true) => {
      if (!content.trim() || isLoading) return;

      setError(null);

      // Add user message
      const userMessage: Message = {
        id: generateId(),
        role: 'user',
        content,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);

      // Build history for context
      const history: ApiChatMessage[] = messages.slice(-10).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      try {
        if (stream) {
          // Streaming response
          const assistantId = generateId();
          let fullContent = '';

          // Add empty assistant message
          setMessages((prev) => [
            ...prev,
            {
              id: assistantId,
              role: 'assistant',
              content: '',
              timestamp: new Date(),
            },
          ]);

          for await (const chunk of chatApi.streamMessage({
            message: content,
            history,
            use_rag: useRag,
            stream: true,
          })) {
            fullContent += chunk;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: fullContent } : m
              )
            );
          }
        } else {
          // Non-streaming response
          const response = await chatApi.sendMessage({
            message: content,
            history,
            use_rag: useRag,
            stream: false,
          });

          const assistantMessage: Message = {
            id: generateId(),
            role: 'assistant',
            content: response.response,
            timestamp: new Date(),
            sources: response.sources?.map((s) => ({
              content: s.content,
              source: s.source,
            })),
          };
          setMessages((prev) => [...prev, assistantMessage]);
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : '发送消息失败';
        setError(errorMessage);
        console.error('Chat error:', err);
      } finally {
        setIsLoading(false);
      }
    },
    [messages, isLoading, useRag]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
  }, []);

  return {
    messages,
    isLoading,
    error,
    useRag,
    setUseRag,
    sendMessage,
    clearMessages,
    stopGeneration,
  };
}

export default useChat;
