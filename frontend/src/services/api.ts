/**
 * API Service for Intelligent Customer Service
 */
import axios from 'axios';

const API_BASE_URL = '/api';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  history?: ChatMessage[];
  use_rag?: boolean;
  stream?: boolean;
}

export interface ChatResponse {
  response: string;
  conversation_id?: string;
  sources?: Array<{
    content: string;
    source: string;
    metadata: Record<string, unknown>;
  }>;
}

export interface KnowledgeAddRequest {
  text: string;
  title?: string;
  metadata?: Record<string, unknown>;
}

export interface KnowledgeResponse {
  success: boolean;
  message: string;
  num_documents?: number;
  details?: Record<string, unknown>;
}

export interface SearchResult {
  content: string;
  source: string;
  metadata: Record<string, unknown>;
}

export interface SystemInfo {
  app_name: string;
  version: string;
  llm_info: {
    provider: string;
    model: string;
    temperature: number;
  };
  status: string;
}

// Agent types
export interface AgentChatRequest {
  message: string;
  conversation_id?: string;
  user_id?: string;  // For personalized long-term memory
  history?: ChatMessage[];
  stream?: boolean;
}

// LangGraph Agent types
export interface LangGraphChatRequest {
  message: string;
  conversation_id?: string;
  user_id?: string;  // For personalized long-term memory
  history?: ChatMessage[];
  stream?: boolean;
  enable_planning?: boolean;
}

export interface LangGraphChatResponse {
  response: string;
  conversation_id?: string;
  plan?: Array<{ id: number; description: string; status: string }>;
  thoughts?: AgentThought[];
  tool_calls?: AgentToolCall[];
  iterations: number;
  parallel_executions?: number;
  error_recoveries?: number;
  error?: string;
}

// Conversation History types
export interface ConversationSummary {
  conversation_id: string;
  title?: string;
  message_count: number;
  interaction_count: number;
  has_summary: boolean;
  created_at: string;
  updated_at: string;
  last_message?: { role: string; content: string };
}

export interface ConversationDetail {
  conversation_id: string;
  title?: string;
  messages: Array<{ role: string; content: string; timestamp?: string }>;
  interactions: Array<{
    interaction_id: string;
    timestamp: string;
    question: string;
    response: string;
    thoughts: AgentThought[];
    tool_calls: AgentToolCall[];
    iterations: number;
    duration_ms: number;
    error?: string;
  }>;
  summary?: string;
  message_count: number;
  interaction_count: number;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

// Long-term Memory types
export interface UserPreference {
  key: string;
  value: unknown;
  category?: string;
  updated_at?: string;
}

export interface MemoryEntity {
  entity_id: string;
  entity_type: string;
  name: string;
  attributes: Record<string, unknown>;
  relationships?: Array<{ type: string; target: string }>;
}

export interface MemoryKnowledge {
  key?: string;
  topic: string;
  content: string;
  source?: string;
  tags?: string[];
  relevance_score?: number;
}

export interface MemoryStats {
  store_type: string;
  namespace_count: number;
  total_items: number;
  namespaces: string[];
}

export interface AgentThought {
  step: number;
  thought: string;
  action?: string;
  action_input?: Record<string, unknown>;
  observation?: string;
}

export interface AgentToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result?: unknown;
  error?: string;
  duration_ms?: number;
}

export interface AgentChatResponse {
  response: string;
  conversation_id?: string;
  thoughts?: AgentThought[];
  tool_calls?: AgentToolCall[];
  iterations: number;
  error?: string;
}

export interface AgentStreamEvent {
  type: 'status' | 'thought' | 'action' | 'observation' | 'response_start' | 'response' | 'response_end' | 'done' | 'error';
  content?: string;
  step?: number;
  tool?: string;
  input?: Record<string, unknown>;
  success?: boolean;
  iterations?: number;
  tool_calls?: number;
}

export interface MCPTool {
  name: string;
  description: string;
  parameters: Array<{
    name: string;
    type: string;
    description: string;
    required: boolean;
  }>;
}

// Model Configuration types
export interface ProviderInfo {
  id: string;
  name: string;
  models: string[];
  requires_api_key: boolean;
  base_url: string;
}

export interface ModelConfig {
  provider: string;
  model: string;
  base_url: string;
  temperature: number;
  max_tokens: number;
  supports_tools: boolean;
  extended_thinking?: boolean;
}

export interface ModelConfigRequest {
  provider: string;
  model: string;
  api_key?: string;
  base_url?: string;
  temperature?: number;
  max_tokens?: number;
}

export interface ValidateConfigRequest {
  provider: string;
  api_key?: string;
}

export interface ValidateConfigResponse {
  valid: boolean;
  error?: string;
  provider?: string;
  name?: string;
}

export interface TestConfigResponse {
  success: boolean;
  provider: string;
  model: string;
  response?: string;
  error?: string;
}

// API client instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Chat API
export const chatApi = {
  /**
   * Send a chat message
   */
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await apiClient.post<ChatResponse>('/chat/message', request);
    return response.data;
  },

  /**
   * Stream a chat message
   */
  streamMessage: async function* (request: ChatRequest): AsyncGenerator<string> {
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ...request, stream: true }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error('Response body is null');
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            return;
          }
          yield data;
        }
      }
    }
  },
};

// Knowledge Base API
export const knowledgeApi = {
  /**
   * Upload a document
   */
  uploadDocument: async (file: File, title?: string): Promise<KnowledgeResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    if (title) {
      formData.append('title', title);
    }

    const response = await apiClient.post<KnowledgeResponse>('/knowledge/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * Add text to knowledge base
   */
  addText: async (request: KnowledgeAddRequest): Promise<KnowledgeResponse> => {
    const response = await apiClient.post<KnowledgeResponse>('/knowledge/add-text', request);
    return response.data;
  },

  /**
   * Search knowledge base
   */
  search: async (query: string, topK: number = 5): Promise<SearchResult[]> => {
    const response = await apiClient.post<{ results: SearchResult[]; query: string }>(
      '/knowledge/search',
      { query, top_k: topK }
    );
    return response.data.results;
  },

  /**
   * Clear knowledge base
   */
  clear: async (): Promise<{ success: boolean; message: string }> => {
    const response = await apiClient.delete('/knowledge/clear');
    return response.data;
  },
};

// Agent API
export const agentApi = {
  /**
   * Send an agent chat message
   */
  sendMessage: async (request: AgentChatRequest): Promise<AgentChatResponse> => {
    const response = await apiClient.post<AgentChatResponse>('/agent/chat', request);
    return response.data;
  },

  /**
   * Stream an agent chat with intermediate steps
   */
  streamMessage: async function* (request: AgentChatRequest): AsyncGenerator<AgentStreamEvent> {
    const response = await fetch(`${API_BASE_URL}/agent/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ...request, stream: true }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error('Response body is null');
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            return;
          }
          try {
            const event = JSON.parse(data) as AgentStreamEvent;
            yield event;
          } catch {
            // Not JSON, skip
          }
        }
      }
    }
  },

  /**
   * Get agent memory for a conversation
   */
  getMemory: async (conversationId: string) => {
    const response = await apiClient.get(`/agent/memory/${conversationId}`);
    return response.data;
  },

  /**
   * Clear agent memory
   */
  clearMemory: async (conversationId: string) => {
    const response = await apiClient.delete(`/agent/memory/${conversationId}`);
    return response.data;
  },
};

// MCP Tools API
export const mcpApi = {
  /**
   * List all available MCP tools
   */
  listTools: async (): Promise<MCPTool[]> => {
    const response = await apiClient.get<MCPTool[]>('/mcp/tools');
    return response.data;
  },

  /**
   * Get tool information
   */
  getToolInfo: async (toolName: string) => {
    const response = await apiClient.get(`/mcp/tools/${toolName}`);
    return response.data;
  },

  /**
   * Execute a tool
   */
  executeTool: async (toolName: string, params: Record<string, unknown>) => {
    const response = await apiClient.post(`/mcp/tools/${toolName}/execute`, { params });
    return response.data;
  },
};

// System API
export const systemApi = {
  /**
   * Get system info
   */
  getInfo: async (): Promise<SystemInfo> => {
    const response = await apiClient.get<SystemInfo>('/system/info');
    return response.data;
  },

  /**
   * Health check
   */
  healthCheck: async (): Promise<{ status: string; timestamp: string }> => {
    const response = await apiClient.get('/system/health');
    return response.data;
  },

  /**
   * Get configuration
   */
  getConfig: async (): Promise<Record<string, unknown>> => {
    const response = await apiClient.get('/system/config');
    return response.data;
  },
};

// Model Configuration API
export const configApi = {
  /**
   * Get all available LLM providers
   */
  getProviders: async (): Promise<ProviderInfo[]> => {
    const response = await apiClient.get<ProviderInfo[]>('/config/providers');
    return response.data;
  },

  /**
   * Get available models for a provider
   */
  getProviderModels: async (providerId: string): Promise<{ provider: string; models: string[] }> => {
    const response = await apiClient.get(`/config/providers/${providerId}/models`);
    return response.data;
  },

  /**
   * Get current model configuration
   */
  getCurrentConfig: async (): Promise<ModelConfig> => {
    const response = await apiClient.get<ModelConfig>('/config/current');
    return response.data;
  },

  /**
   * Validate model configuration
   */
  validateConfig: async (request: ValidateConfigRequest): Promise<ValidateConfigResponse> => {
    const response = await apiClient.post<ValidateConfigResponse>('/config/validate', request);
    return response.data;
  },

  /**
   * Update model configuration
   */
  updateConfig: async (request: ModelConfigRequest): Promise<ModelConfig> => {
    const response = await apiClient.post<ModelConfig>('/config/update', request);
    return response.data;
  },

  /**
   * Test model configuration without applying it
   */
  testConfig: async (request: ModelConfigRequest): Promise<TestConfigResponse> => {
    const response = await apiClient.post<TestConfigResponse>('/config/test', request);
    return response.data;
  },
};

// LangGraph Agent API
export const langgraphApi = {
  /**
   * Send a LangGraph agent chat message
   */
  sendMessage: async (request: LangGraphChatRequest): Promise<LangGraphChatResponse> => {
    const response = await apiClient.post<LangGraphChatResponse>('/agent/langgraph/chat', request);
    return response.data;
  },

  /**
   * Stream a LangGraph agent chat with intermediate steps
   */
  streamMessage: async function* (request: LangGraphChatRequest): AsyncGenerator<AgentStreamEvent> {
    const response = await fetch(`${API_BASE_URL}/agent/langgraph/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ...request, stream: true }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error('Response body is null');
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            return;
          }
          try {
            const event = JSON.parse(data) as AgentStreamEvent;
            yield event;
          } catch {
            // Not JSON, skip
          }
        }
      }
    }
  },

  /**
   * Get agent capabilities
   */
  getCapabilities: async () => {
    const response = await apiClient.get('/agent/capabilities');
    return response.data;
  },
};

// Conversation History API
export const conversationApi = {
  /**
   * List all conversations
   */
  list: async (params?: { limit?: number; offset?: number; sort_by?: string; descending?: boolean }): Promise<{
    conversations: ConversationSummary[];
    total: number;
    limit: number;
    offset: number;
  }> => {
    const response = await apiClient.get('/agent/conversations', { params });
    return response.data;
  },

  /**
   * Get conversation detail
   */
  getDetail: async (conversationId: string): Promise<ConversationDetail> => {
    const response = await apiClient.get(`/agent/conversations/${conversationId}`);
    return response.data;
  },

  /**
   * Update conversation (e.g., title)
   */
  update: async (conversationId: string, data: { title?: string }): Promise<{ success: boolean }> => {
    const response = await apiClient.patch(`/agent/conversations/${conversationId}`, data);
    return response.data;
  },

  /**
   * Delete conversation
   */
  delete: async (conversationId: string): Promise<{ success: boolean }> => {
    const response = await apiClient.delete(`/agent/conversations/${conversationId}`);
    return response.data;
  },

  /**
   * Export conversation
   */
  export: async (conversationId: string): Promise<{ conversation_id: string; data: unknown; exported_at: string }> => {
    const response = await apiClient.post(`/agent/conversations/${conversationId}/export`);
    return response.data;
  },

  /**
   * Import conversation
   */
  import: async (data: unknown): Promise<{ success: boolean; conversation_id: string }> => {
    const response = await apiClient.post('/agent/conversations/import', data);
    return response.data;
  },
};

// Long-term Memory Store API
export const memoryStoreApi = {
  /**
   * Get memory store statistics
   */
  getStats: async (): Promise<MemoryStats> => {
    const response = await apiClient.get<MemoryStats>('/agent/store/stats');
    return response.data;
  },

  // User Preferences
  /**
   * Set user preference
   */
  setUserPreference: async (userId: string, preference: { key: string; value: unknown; category?: string }): Promise<UserPreference> => {
    const response = await apiClient.post(`/agent/store/users/${userId}/preferences`, preference);
    return response.data;
  },

  /**
   * Get all user preferences
   */
  getUserPreferences: async (userId: string, category?: string): Promise<{ user_id: string; preferences: UserPreference[]; category?: string }> => {
    const response = await apiClient.get(`/agent/store/users/${userId}/preferences`, { params: { category } });
    return response.data;
  },

  /**
   * Get specific user preference
   */
  getUserPreference: async (userId: string, key: string): Promise<{ user_id: string; key: string; value: unknown }> => {
    const response = await apiClient.get(`/agent/store/users/${userId}/preferences/${key}`);
    return response.data;
  },

  // Entities
  /**
   * Store entity
   */
  storeEntity: async (entity: { entity_id: string; entity_type: string; name: string; attributes?: Record<string, unknown>; relationships?: Array<{ type: string; target: string }> }): Promise<MemoryEntity> => {
    const response = await apiClient.post('/agent/store/entities', entity);
    return response.data;
  },

  /**
   * Get entity
   */
  getEntity: async (entityType: string, entityId: string): Promise<MemoryEntity> => {
    const response = await apiClient.get(`/agent/store/entities/${entityType}/${entityId}`);
    return response.data;
  },

  /**
   * Search entities
   */
  searchEntities: async (query: string, entityType?: string, limit?: number): Promise<{ query: string; results: MemoryEntity[]; count: number }> => {
    const response = await apiClient.get('/agent/store/entities/search', {
      params: { query, entity_type: entityType, limit },
    });
    return response.data;
  },

  // Knowledge
  /**
   * Store knowledge
   */
  storeKnowledge: async (knowledge: { topic: string; content: string; source?: string; tags?: string[] }): Promise<MemoryKnowledge> => {
    const response = await apiClient.post('/agent/store/knowledge', knowledge);
    return response.data;
  },

  /**
   * Search knowledge
   */
  searchKnowledge: async (query: string, limit?: number): Promise<{ results: MemoryKnowledge[]; total: number; query: string }> => {
    const response = await apiClient.post('/agent/store/knowledge/search', { query, limit });
    return response.data;
  },

  // User Context
  /**
   * Get user context (preferences + entities)
   */
  getUserContext: async (userId: string, options?: { include_preferences?: boolean; include_entities?: boolean }): Promise<Record<string, unknown>> => {
    const response = await apiClient.get(`/agent/store/users/${userId}/context`, { params: options });
    return response.data;
  },

  /**
   * Clear user memory
   */
  clearUserMemory: async (userId: string): Promise<{ success: boolean; user_id: string; deleted_items: number }> => {
    const response = await apiClient.delete(`/agent/store/users/${userId}`);
    return response.data;
  },
};

export default {
  chat: chatApi,
  knowledge: knowledgeApi,
  system: systemApi,
  agent: agentApi,
  mcp: mcpApi,
  config: configApi,
  langgraph: langgraphApi,
  conversation: conversationApi,
  memoryStore: memoryStoreApi,
};
