import { useState, useCallback } from 'react';
import { chatApi, agentApi, memoryApi, AgentStreamEvent, ChatRequest } from '../services/api';
import { Message } from '../components/ChatMessage';
import { ThinkingStep } from '../components/ThinkingIndicator';

interface UseAgentChatOptions {
  useRag?: boolean;
  agentMode?: boolean;
  useLangGraph?: boolean;
}

export const useAgentChat = (options: UseAgentChatOptions = {}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useRag, setUseRag] = useState(options.useRag ?? true);
  const [agentMode, setAgentMode] = useState(options.agentMode ?? false);
  const [useLangGraph, setUseLangGraph] = useState(options.useLangGraph ?? true);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [thinkingSteps, setThinkingSteps] = useState<ThinkingStep[]>([]);
  const [currentStatus, setCurrentStatus] = useState<string>('');
  const [currentStep, setCurrentStep] = useState<number>(0);

  const addMessage = useCallback((message: Message) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const updateLastMessage = useCallback((content: string) => {
    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const updated = [...prev];
      updated[updated.length - 1] = {
        ...updated[updated.length - 1],
        content,
      };
      return updated;
    });
  }, []);

  const sendMessage = useCallback(
    async (content: string, stream: boolean = true) => {
      if (!content.trim()) return;

      // Add user message
      const userMessage: Message = {
        id: `user_${Date.now()}`,
        role: 'user',
        content,
        timestamp: new Date(),
      };
      addMessage(userMessage);

      setIsLoading(true);
      setError(null);
      setThinkingSteps([]);
      setCurrentStatus('');
      setCurrentStep(0);

      try {
        // Create conversation if needed
        let currentConvId = conversationId;
        if (!currentConvId) {
          const conv = await memoryApi.createConversation();
          currentConvId = conv.conversation_id;
          setConversationId(currentConvId);
        }

        // Add user message to memory
        await memoryApi.addMessage(currentConvId, { role: 'user', content });

        const request: ChatRequest = {
          message: content,
          conversation_id: currentConvId,
          use_rag: useRag,
          stream,
          config: {
            enable_tools: agentMode,
            enable_rag: useRag,
          },
        };

        if (agentMode && stream) {
          // Use agent streaming
          const assistantMessage: Message = {
            id: `assistant_${Date.now()}`,
            role: 'assistant',
            content: '',
            timestamp: new Date(),
          };
          addMessage(assistantMessage);

          let fullContent = '';
          for await (const event of agentApi.streamMessage(request)) {
            handleAgentEvent(event, (text) => {
              fullContent += text;
              updateLastMessage(fullContent);
            });
          }

          // Save assistant message to memory
          if (fullContent) {
            await memoryApi.addMessage(currentConvId, { role: 'assistant', content: fullContent });
          }
        } else if (stream) {
          // Use regular chat streaming
          const assistantMessage: Message = {
            id: `assistant_${Date.now()}`,
            role: 'assistant',
            content: '',
            timestamp: new Date(),
          };
          addMessage(assistantMessage);

          let fullContent = '';
          for await (const chunk of chatApi.streamMessage(request)) {
            try {
              const data = JSON.parse(chunk);
              if (data.content) {
                fullContent += data.content;
                updateLastMessage(fullContent);
              }
            } catch {
              // Plain text chunk
              fullContent += chunk;
              updateLastMessage(fullContent);
            }
          }

          // Save assistant message to memory
          if (fullContent) {
            await memoryApi.addMessage(currentConvId, { role: 'assistant', content: fullContent });
          }
        } else {
          // Non-streaming
          const response = await chatApi.sendMessage(request);
          const assistantMessage: Message = {
            id: `assistant_${Date.now()}`,
            role: 'assistant',
            content: response.message,
            timestamp: new Date(),
            sources: response.sources?.map((s) => ({ content: s, source: s })),
            toolCalls: response.tool_calls,
          };
          addMessage(assistantMessage);

          // Save assistant message to memory
          await memoryApi.addMessage(currentConvId, { role: 'assistant', content: response.message });
        }
      } catch (err) {
        console.error('Chat error:', err);
        setError(err instanceof Error ? err.message : 'Failed to send message');
      } finally {
        setIsLoading(false);
        setCurrentStatus('');
      }
    },
    [conversationId, useRag, agentMode, addMessage, updateLastMessage]
  );

  const handleAgentEvent = (event: AgentStreamEvent, onContent: (text: string) => void) => {
    switch (event.type) {
      case 'status':
        setCurrentStatus(event.content || '');
        break;
      case 'thought':
        if (event.content) {
          setThinkingSteps((prev) => [
            ...prev,
            { type: 'thinking', content: event.content!, step: event.step },
          ]);
          if (event.step !== undefined) setCurrentStep(event.step);
        }
        break;
      case 'action':
        if (event.tool) {
          setThinkingSteps((prev) => [
            ...prev,
            { type: 'action', content: event.tool!, tool: event.tool, step: event.step },
          ]);
        }
        break;
      case 'observation':
        if (event.content) {
          setThinkingSteps((prev) => [
            ...prev,
            { type: 'observation', content: event.content!, step: event.step },
          ]);
        }
        break;
      case 'response':
      case 'response_start':
        if (event.content) {
          onContent(event.content);
        }
        break;
      case 'done':
        setCurrentStatus('完成');
        break;
      case 'error':
        setError(event.content || 'Unknown error');
        break;
    }
  };

  const clearMessages = useCallback(() => {
    setMessages([]);
    setThinkingSteps([]);
    setCurrentStatus('');
    setCurrentStep(0);
    setError(null);
  }, []);

  const startNewConversation = useCallback(() => {
    setConversationId(undefined);
    clearMessages();
  }, [clearMessages]);

  const loadConversation = useCallback((convId: string, loadedMessages: Message[]) => {
    setConversationId(convId);
    setMessages(loadedMessages);
  }, []);

  return {
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
  };
};

export default useAgentChat;
