import { useState, useCallback, useRef } from 'react';
import { a2aApi, A2ARoutingMode, A2AStreamEvent } from '../services/api';
import { Message } from '../components/ChatMessage';

export interface MultiAgentMessage extends Message {
  agentName?: string;
  agentDisplayName?: string;
  routingMode?: A2ARoutingMode;
  isMultiAgentResponse?: boolean;
  parallelResults?: Array<{
    agent: string;
    agentDisplayName: string;
    response: string;
  }>;
  chainSteps?: Array<{
    agent: string;
    agentDisplayName: string;
    response: string;
    step: number;
  }>;
}

export interface UseMultiAgentChatOptions {
  conversationId?: string;
  onConversationCreated?: (conversationId: string) => void;
}

const AGENT_DISPLAY_NAMES: Record<string, string> = {
  travel_assistant: '出行助手',
  charging_manager: '充电管家',
  billing_advisor: '费用顾问',
  emergency_support: '故障急救',
  data_analyst: '数据分析师',
  maintenance_expert: '运维专家',
  energy_advisor: '能源顾问',
  scheduling_advisor: '调度顾问',
};

const getAgentDisplayName = (agentName: string): string => {
  return AGENT_DISPLAY_NAMES[agentName] || agentName;
};

export function useMultiAgentChat(options: UseMultiAgentChatOptions = {}) {
  const [messages, setMessages] = useState<MultiAgentMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState(options.conversationId ?? '');
  const [currentAgent, setCurrentAgent] = useState<string>('');
  const [currentStatus, setCurrentStatus] = useState<string>('');

  const onConversationCreatedRef = useRef(options.onConversationCreated);
  onConversationCreatedRef.current = options.onConversationCreated;

  const generateId = () => `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  const generateConversationId = () => `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  const sendMessage = useCallback(
    async (
      content: string,
      routingMode: A2ARoutingMode = 'auto',
      selectedAgents: string[] = [],
      stream: boolean = false  // Disabled streaming as backend doesn't support /a2a/chat/stream yet
    ) => {
      if (!content.trim() || isLoading) return;

      setError(null);
      setCurrentAgent('');
      setCurrentStatus('正在路由请求...');

      // Generate conversation ID if not exists
      let currentConvId = conversationId;
      if (!currentConvId) {
        currentConvId = generateConversationId();
        setConversationId(currentConvId);
        onConversationCreatedRef.current?.(currentConvId);
      }

      // Add user message
      const userMessage: MultiAgentMessage = {
        id: generateId(),
        role: 'user',
        content,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);

      try {
        if (stream && routingMode === 'auto') {
          // Streaming for AUTO mode
          const assistantId = generateId();
          let fullContent = '';
          let respondingAgent = '';

          // Add empty assistant message
          setMessages((prev) => [
            ...prev,
            {
              id: assistantId,
              role: 'assistant',
              content: '',
              timestamp: new Date(),
              isMultiAgentResponse: true,
              routingMode: 'auto',
            },
          ]);

          for await (const event of a2aApi.streamMessage({
            message: content,
            mode: routingMode,
            conversation_id: currentConvId,
          })) {
            handleStreamEvent(event, assistantId, fullContent, respondingAgent, (newContent) => {
              fullContent = newContent;
            }, (agent) => {
              respondingAgent = agent;
            });
          }

          // Final update
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    content: fullContent,
                    agentName: respondingAgent,
                    agentDisplayName: getAgentDisplayName(respondingAgent),
                  }
                : m
            )
          );
        } else if (routingMode === 'parallel') {
          // Parallel mode - call multiple agents simultaneously
          setCurrentStatus('正在并行调用智能体...');

          const response = await a2aApi.sendParallel({
            message: content,
            agents: selectedAgents,
            conversation_id: currentConvId,
          });

          // Backend returns results as a Record, convert to array
          const parallelResults = response.results
            ? Object.entries(response.results).map(([agentName, r]) => ({
                agent: agentName,
                agentDisplayName: getAgentDisplayName(agentName),
                response: (r as { response_text?: string })?.response_text || '',
              }))
            : [];

          // Combine all responses
          const combinedContent = parallelResults
            .map((r) => `**${r.agentDisplayName}**:\n${r.response}`)
            .join('\n\n---\n\n');

          const assistantMessage: MultiAgentMessage = {
            id: generateId(),
            role: 'assistant',
            content: combinedContent,
            timestamp: new Date(),
            isMultiAgentResponse: true,
            routingMode: 'parallel',
            parallelResults,
          };
          setMessages((prev) => [...prev, assistantMessage]);
        } else if (routingMode === 'sequential') {
          // Sequential mode - chain agents
          setCurrentStatus('正在执行智能体链...');

          const response = await a2aApi.sendSequential({
            message: content,
            agents: selectedAgents,
            conversation_id: currentConvId,
          });

          // Backend returns steps with result field
          const chainSteps = response.steps?.map((s) => ({
            agent: s.agent,
            agentDisplayName: getAgentDisplayName(s.agent),
            response: s.result || '',
            step: s.step,
          })) || [];

          // Use final_result or combine chain
          const finalContent = response.final_result || chainSteps
            .map((s) => `**步骤 ${s.step} - ${s.agentDisplayName}**:\n${s.response}`)
            .join('\n\n---\n\n');

          const assistantMessage: MultiAgentMessage = {
            id: generateId(),
            role: 'assistant',
            content: finalContent,
            timestamp: new Date(),
            isMultiAgentResponse: true,
            routingMode: 'sequential',
            chainSteps,
          };
          setMessages((prev) => [...prev, assistantMessage]);
        } else {
          // Non-streaming AUTO mode
          const response = await a2aApi.sendMessage({
            message: content,
            mode: routingMode,
            conversation_id: currentConvId,
          });

          const assistantMessage: MultiAgentMessage = {
            id: generateId(),
            role: 'assistant',
            content: response.response_text || response.response || '',
            timestamp: new Date(),
            isMultiAgentResponse: true,
            routingMode: 'auto',
            agentName: response.agent,
            agentDisplayName: response.agent ? getAgentDisplayName(response.agent) : undefined,
          };
          setMessages((prev) => [...prev, assistantMessage]);
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : '发送消息失败';
        setError(errorMessage);
        console.error('Multi-agent chat error:', err);
      } finally {
        setIsLoading(false);
        setCurrentStatus('');
        setCurrentAgent('');
      }
    },
    [conversationId, isLoading]
  );

  const handleStreamEvent = (
    event: A2AStreamEvent,
    assistantId: string,
    currentContent: string,
    _currentAgent: string,
    setContent: (content: string) => void,
    setAgent: (agent: string) => void
  ) => {
    switch (event.type) {
      case 'status':
        setCurrentStatus(event.content || '');
        break;

      case 'agent_start':
        if (event.agent) {
          setAgent(event.agent);
          setCurrentAgent(event.agent);
          setCurrentStatus(`${getAgentDisplayName(event.agent)} 正在处理...`);
        }
        break;

      case 'agent_response':
      case 'response':
        const newContent = currentContent + (event.content || '');
        setContent(newContent);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: newContent,
                  agentName: event.agent || _currentAgent,
                  agentDisplayName: getAgentDisplayName(event.agent || _currentAgent),
                }
              : m
          )
        );
        break;

      case 'agent_done':
        setCurrentStatus(`${getAgentDisplayName(event.agent || '')} 完成`);
        break;

      case 'done':
        setCurrentStatus('');
        break;

      case 'error':
        setError(event.content || '未知错误');
        break;
    }
  };

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
    setCurrentAgent('');
    setCurrentStatus('');
  }, []);

  const startNewConversation = useCallback(() => {
    clearMessages();
    const newConvId = generateConversationId();
    setConversationId(newConvId);
    onConversationCreatedRef.current?.(newConvId);
    return newConvId;
  }, [clearMessages]);

  const loadConversation = useCallback((convId: string, loadedMessages: MultiAgentMessage[]) => {
    setConversationId(convId);
    setMessages(loadedMessages);
    setCurrentAgent('');
    setCurrentStatus('');
  }, []);

  return {
    messages,
    isLoading,
    error,
    conversationId,
    currentAgent,
    currentStatus,
    sendMessage,
    clearMessages,
    startNewConversation,
    loadConversation,
    setConversationId,
  };
}

export default useMultiAgentChat;
