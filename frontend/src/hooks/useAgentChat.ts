import { useState, useCallback, useRef } from 'react';
import {
  agentApi,
  chatApi,
  langgraphApi,
  ChatMessage as ApiChatMessage,
  AgentStreamEvent,
  AgentToolCall
} from '../services/api';
import { Message } from '../components/ChatMessage';
import { ThinkingStep } from '../components/ThinkingIndicator';

export interface UseAgentChatOptions {
  useRag?: boolean;
  agentMode?: boolean;
  useLangGraph?: boolean;
  userId?: string;
  conversationId?: string;
  onConversationCreated?: (conversationId: string) => void;
}

export function useAgentChat(options: UseAgentChatOptions = {}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useRag, setUseRag] = useState(options.useRag ?? true);
  const [agentMode, setAgentMode] = useState(options.agentMode ?? false);
  const [useLangGraph, setUseLangGraph] = useState(options.useLangGraph ?? true);
  const [userId, setUserId] = useState(options.userId ?? '');
  const [conversationId, setConversationId] = useState(options.conversationId ?? '');

  // Agent-specific state
  const [thinkingSteps, setThinkingSteps] = useState<ThinkingStep[]>([]);
  const [currentStatus, setCurrentStatus] = useState<string>('');
  const [currentStep, setCurrentStep] = useState(0);

  // Plan state for LangGraph
  const [currentPlan, setCurrentPlan] = useState<Array<{ id: number; description: string; status: string }>>([]);

  const onConversationCreatedRef = useRef(options.onConversationCreated);
  onConversationCreatedRef.current = options.onConversationCreated;

  const generateId = () => `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  const generateConversationId = () => `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  const sendMessage = useCallback(
    async (content: string, stream: boolean = true) => {
      if (!content.trim() || isLoading) return;

      setError(null);
      setThinkingSteps([]);
      setCurrentStatus('');
      setCurrentStep(0);
      setCurrentPlan([]);

      // Generate conversation ID if not exists
      let currentConvId = conversationId;
      if (!currentConvId) {
        currentConvId = generateConversationId();
        setConversationId(currentConvId);
        onConversationCreatedRef.current?.(currentConvId);
      }

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
        if (agentMode) {
          // Agent mode with streaming
          const assistantId = generateId();
          let fullContent = '';
          const toolCalls: AgentToolCall[] = [];
          let iterations = 0;

          // Add empty assistant message
          setMessages((prev) => [
            ...prev,
            {
              id: assistantId,
              role: 'assistant',
              content: '',
              timestamp: new Date(),
              isAgentResponse: true,
            },
          ]);

          // Choose API based on useLangGraph setting
          const streamApi = useLangGraph ? langgraphApi : agentApi;
          const request = {
            message: content,
            conversation_id: currentConvId,
            user_id: userId || undefined,
            history,
            stream: true,
          };

          for await (const event of streamApi.streamMessage(request)) {
            handleAgentEvent(event, assistantId, fullContent, toolCalls, iterations, (newContent) => {
              fullContent = newContent;
            }, (newIterations) => {
              iterations = newIterations;
            }, (newToolCalls) => {
              toolCalls.push(...newToolCalls);
            });
          }

          // Final update with all tool calls
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: fullContent, toolCalls, iterations, isAgentResponse: true }
                : m
            )
          );

        } else if (stream) {
          // Normal streaming response
          const assistantId = generateId();
          let fullContent = '';

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
            conversation_id: currentConvId,
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
            conversation_id: currentConvId,
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
        setCurrentStatus('');
      }
    },
    [messages, isLoading, useRag, agentMode, useLangGraph, userId, conversationId]
  );

  const handleAgentEvent = (
    event: AgentStreamEvent,
    assistantId: string,
    currentContent: string,
    _toolCalls: AgentToolCall[],
    _iterations: number,
    setContent: (content: string) => void,
    setIterations: (iterations: number) => void,
    _addToolCalls: (calls: AgentToolCall[]) => void
  ) => {
    switch (event.type) {
      case 'status':
        setCurrentStatus(event.content || '');
        break;

      case 'thought':
        setCurrentStep(event.step || 0);
        setThinkingSteps((prev) => [
          ...prev,
          {
            type: 'thinking',
            content: event.content || '',
            step: event.step,
          },
        ]);
        break;

      case 'action':
        setThinkingSteps((prev) => [
          ...prev,
          {
            type: 'action',
            content: `调用工具: ${event.tool}`,
            tool: event.tool,
            step: event.step,
          },
        ]);
        break;

      case 'observation':
        setThinkingSteps((prev) => [
          ...prev,
          {
            type: 'observation',
            content: event.content || '',
            step: event.step,
          },
        ]);
        break;

      case 'response':
        const newContent = currentContent + (event.content || '');
        setContent(newContent);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: newContent } : m
          )
        );
        break;

      case 'done':
        setIterations(event.iterations || 0);
        break;

      case 'error':
        setError(event.content || '未知错误');
        break;
    }
  };

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
    setThinkingSteps([]);
    setCurrentStatus('');
    setCurrentStep(0);
    setCurrentPlan([]);
  }, []);

  const startNewConversation = useCallback(() => {
    clearMessages();
    const newConvId = generateConversationId();
    setConversationId(newConvId);
    onConversationCreatedRef.current?.(newConvId);
    return newConvId;
  }, [clearMessages]);

  const loadConversation = useCallback((convId: string, loadedMessages: Message[]) => {
    setConversationId(convId);
    setMessages(loadedMessages);
    setThinkingSteps([]);
    setCurrentStatus('');
    setCurrentStep(0);
    setCurrentPlan([]);
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
    userId,
    setUserId,
    conversationId,
    setConversationId,
    sendMessage,
    clearMessages,
    startNewConversation,
    loadConversation,
    // Agent-specific
    thinkingSteps,
    currentStatus,
    currentStep,
    currentPlan,
  };
}

export default useAgentChat;
