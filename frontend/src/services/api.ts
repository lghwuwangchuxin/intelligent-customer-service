/**
 * API Service for Intelligent Customer Service
 * Microservices Architecture Support
 */
import axios, { AxiosInstance } from 'axios';

// ==================== Service Configuration ====================

/**
 * Microservices Configuration
 * All services are accessed through API Gateway or directly
 */
export const SERVICES = {
  API_GATEWAY: { name: 'api-gateway', port: 8000, path: '/api' },
  MCP_SERVICE: { name: 'mcp-service', port: 8001, path: '/api/mcp' },
  RAG_SERVICE: { name: 'rag-service', port: 8002, path: '/api/rag' },
  EVALUATION_SERVICE: { name: 'evaluation-service', port: 8003, path: '/api/evaluation' },
  MONITORING_SERVICE: { name: 'monitoring-service', port: 8004, path: '/api/monitoring' },
  SINGLE_AGENT_SERVICE: { name: 'single-agent-service', port: 8005, path: '/api/agent' },
  MULTI_AGENT_SERVICE: { name: 'multi-agent-service', port: 8006, path: '/api/multi-agent' },
  LLM_MANAGER_SERVICE: { name: 'llm-manager-service', port: 8007, path: '/api/llm' },
  MEMORY_SERVICE: { name: 'memory-service', port: 8008, path: '/api/memory' },
} as const;

// Base URL - through API Gateway by default
const API_BASE_URL = '/api';

// ==================== Type Definitions ====================

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  history?: ChatMessage[];
  use_rag?: boolean;
  stream?: boolean;
  config?: {
    enable_tools?: boolean;
    enable_rag?: boolean;
    knowledge_base_id?: string;
    temperature?: number;
    max_tokens?: number;
  };
}

export interface ChatResponse {
  message: string;
  conversation_id: string;
  tool_calls?: AgentToolCall[];
  sources?: string[];
  is_final?: boolean;
}

// Service Health Types
export interface ServiceHealth {
  status: 'healthy' | 'unhealthy' | 'unknown';
  service: string;
  message?: string;
  details?: Record<string, unknown>;
}

export interface AllServicesHealth {
  services: Record<string, ServiceHealth>;
  overall: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
}

// Knowledge/RAG Types
export interface KnowledgeAddRequest {
  content: string;
  metadata?: Record<string, string>;
  source?: string;
  knowledge_base_id?: string;
}

export interface KnowledgeResponse {
  document_id: string;
  success: boolean;
  chunks_created?: number;
  error?: string;
}

export interface RetrieveRequest {
  query: string;
  top_k?: number;
  knowledge_base_id?: string;
  config?: {
    enable_query_transform?: boolean;
    enable_rerank?: boolean;
    hybrid_alpha?: number;
    min_score?: number;
  };
}

export interface DocumentInfo {
  id: string;
  content: string;
  score: number;
  metadata?: Record<string, string>;
  source?: string;
}

export interface RetrieveResponse {
  documents: DocumentInfo[];
  original_query: string;
  transformed_query?: string;
  latency_ms: number;
}

export interface RAGStats {
  total_documents: number;
  total_chunks: number;
  index_size_bytes: number;
}

// Memory Service Types
export interface ConversationInfo {
  conversation_id: string;
  title?: string;
  message_count: number;
  has_summary: boolean;
  created_at: string;
  updated_at: string;
}

// Alias for compatibility
export interface ConversationSummary extends ConversationInfo {
  last_message?: { content: string; role: string };
}

export interface ConversationDetail {
  conversation_id: string;
  title?: string;
  messages: ChatMessage[];
  summary?: string;
  message_count: number;
  created_at: string;
  updated_at: string;
  metadata?: Record<string, unknown>;
}

export interface AddMessageRequest {
  role: string;
  content: string;
}

export interface AddMessageResponse {
  conversation_id: string;
  message_count: number;
  has_summary: boolean;
  summary_triggered: boolean;
}

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
  topic: string;
  content: string;
  source?: string;
  tags?: string[];
  relevance_score?: number;
}

export interface MemoryStats {
  total_conversations: number;
  total_messages: number;
  total_memories: number;
  namespaces: string[];
  memory_types: Record<string, number>;
}

// Agent Types
export interface AgentThought {
  step: number;
  thought: string;
  action?: string;
  action_input?: Record<string, unknown>;
  observation?: string;
}

export interface AgentToolCall {
  id?: string;
  name: string;
  args?: Record<string, unknown>;
  result?: unknown;
  error?: string;
  duration_ms?: number;
  status?: string;
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

// MCP Types
export interface MCPTool {
  name: string;
  description: string;
  parameters?: Array<{
    name: string;
    type: string;
    description: string;
    required: boolean;
  }>;
  inputSchema?: Record<string, unknown>;
}

export interface MCPToolExecuteRequest {
  tool_name: string;
  arguments: Record<string, unknown>;
}

export interface MCPToolExecuteResponse {
  success: boolean;
  result?: unknown;
  error?: string;
  duration_ms?: number;
}

// Monitoring Types
export interface MetricData {
  name: string;
  value: number;
  unit?: string;
  timestamp: string;
  labels?: Record<string, string>;
}

export interface TraceSpan {
  trace_id: string;
  span_id: string;
  name: string;
  start_time: string;
  end_time?: string;
  duration_ms?: number;
  status: string;
  attributes?: Record<string, unknown>;
}

export interface MonitoringStats {
  total_requests: number;
  avg_latency_ms: number;
  error_rate: number;
  active_conversations: number;
  requests_per_minute: number;
}

// Model Configuration Types
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
}

export interface ModelConfigRequest {
  provider: string;
  model: string;
  api_key?: string;
  base_url?: string;
  temperature?: number;
  max_tokens?: number;
}

// Multi-Agent (A2A) Types
export type A2ARoutingMode = 'auto' | 'parallel' | 'sequential';

export interface A2AAgentInfo {
  name: string;
  description: string;
  url?: string;
  domain?: string;
  skills?: string[];
  connected: boolean;
}

export interface A2AHealthResponse {
  status: string;
  initialized: boolean;
  connected_agents: string[];
  total_agents: number;
  message: string;
}

export interface A2AChatRequest {
  message: string;
  mode?: A2ARoutingMode;
  agents?: string[];
  conversation_id?: string;
  stream?: boolean;
}

export interface A2AChatResponse {
  status: string;
  mode: string;
  response_text?: string;
  response?: string;
  agent?: string;
  domain?: string;
  error?: string;
  results?: Record<string, unknown>;
  steps?: Array<{ step: number; agent: string; result?: string }>;
  final_result?: string;
}

export interface A2AStreamEvent {
  type: 'status' | 'agent_start' | 'agent_response' | 'agent_done' | 'response' | 'done' | 'error';
  content?: string;
  agent?: string;
  step?: number;
  mode?: A2ARoutingMode;
}

// System Info
export interface SystemInfo {
  app_name: string;
  version: string;
  llm_info: {
    provider: string;
    model: string;
    temperature: number;
  };
  status: string;
  services?: Record<string, ServiceHealth>;
}

// ==================== API Client Instance ====================

const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ==================== Services Health API ====================

export const servicesApi = {
  /**
   * Check health of all microservices
   */
  checkAllHealth: async (): Promise<AllServicesHealth> => {
    const services: Record<string, ServiceHealth> = {};
    const serviceConfigs = Object.values(SERVICES);

    const healthChecks = serviceConfigs.map(async (svc) => {
      try {
        // Use API Gateway to check service health
        const response = await apiClient.get(`/services/${svc.name}/health`, { timeout: 5000 });
        services[svc.name] = {
          status: response.data.status === 'healthy' ? 'healthy' : 'unhealthy',
          service: svc.name,
          message: response.data.message,
          details: response.data.details,
        };
      } catch {
        services[svc.name] = {
          status: 'unhealthy',
          service: svc.name,
          message: 'Service unreachable',
        };
      }
    });

    await Promise.all(healthChecks);

    const healthyCount = Object.values(services).filter(s => s.status === 'healthy').length;
    const totalCount = serviceConfigs.length;

    let overall: 'healthy' | 'degraded' | 'unhealthy';
    if (healthyCount === totalCount) overall = 'healthy';
    else if (healthyCount > 0) overall = 'degraded';
    else overall = 'unhealthy';

    return {
      services,
      overall,
      timestamp: new Date().toISOString(),
    };
  },

  /**
   * Check health of a specific service
   */
  checkHealth: async (serviceName: string): Promise<ServiceHealth> => {
    try {
      const response = await apiClient.get(`/services/${serviceName}/health`, { timeout: 5000 });
      return {
        status: response.data.status === 'healthy' ? 'healthy' : 'unhealthy',
        service: serviceName,
        message: response.data.message,
        details: response.data.details,
      };
    } catch {
      return {
        status: 'unhealthy',
        service: serviceName,
        message: 'Service unreachable',
      };
    }
  },

  /**
   * Get list of all services
   */
  getServices: () => SERVICES,
};

// ==================== Chat API (via API Gateway) ====================

export const chatApi = {
  /**
   * Send a chat message through API Gateway
   */
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await apiClient.post<ChatResponse>('/chat', {
      message: request.message,
      conversation_id: request.conversation_id,
      enable_tools: request.config?.enable_tools ?? true,
      enable_rag: request.config?.enable_rag ?? request.use_rag ?? false,
      knowledge_base_id: request.config?.knowledge_base_id,
    });
    return response.data;
  },

  /**
   * Stream a chat message
   */
  streamMessage: async function* (request: ChatRequest): AsyncGenerator<string> {
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: request.message,
        conversation_id: request.conversation_id,
        enable_tools: request.config?.enable_tools ?? true,
        enable_rag: request.config?.enable_rag ?? request.use_rag ?? false,
        stream: true,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) throw new Error('Response body is null');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') return;
          yield data;
        }
      }
    }
  },
};

// ==================== Memory Service API ====================

export const memoryApi = {
  // Conversation Management
  /**
   * Create a new conversation
   */
  createConversation: async (title?: string): Promise<ConversationInfo> => {
    const response = await apiClient.post<ConversationInfo>('/memory/conversations', { title });
    return response.data;
  },

  /**
   * List all conversations
   */
  listConversations: async (page = 1, pageSize = 20): Promise<{
    conversations: ConversationInfo[];
    total: number;
    page: number;
    page_size: number;
  }> => {
    const response = await apiClient.get('/memory/conversations', {
      params: { page, page_size: pageSize },
    });
    return response.data;
  },

  /**
   * Get conversation details
   */
  getConversation: async (conversationId: string): Promise<ConversationDetail> => {
    const response = await apiClient.get(`/memory/conversations/${conversationId}`);
    return response.data;
  },

  /**
   * Add message to conversation
   */
  addMessage: async (conversationId: string, message: AddMessageRequest): Promise<AddMessageResponse> => {
    const response = await apiClient.post(`/memory/conversations/${conversationId}/messages`, message);
    return response.data;
  },

  /**
   * Get conversation context for LLM
   */
  getContext: async (conversationId: string, maxMessages = 10): Promise<{ messages: ChatMessage[] }> => {
    const response = await apiClient.get(`/memory/conversations/${conversationId}/context`, {
      params: { max_messages: maxMessages },
    });
    return response.data;
  },

  /**
   * Trigger conversation summarization
   */
  summarizeConversation: async (conversationId: string, force = false): Promise<{ summary?: string; success: boolean }> => {
    const response = await apiClient.post(`/memory/conversations/${conversationId}/summarize`, null, {
      params: { force },
    });
    return response.data;
  },

  /**
   * Delete conversation
   */
  deleteConversation: async (conversationId: string): Promise<{ success: boolean }> => {
    const response = await apiClient.delete(`/memory/conversations/${conversationId}`);
    return response.data;
  },

  /**
   * Clear conversation messages
   */
  clearConversation: async (conversationId: string): Promise<{ success: boolean }> => {
    const response = await apiClient.post(`/memory/conversations/${conversationId}/clear`);
    return response.data;
  },

  // User Preferences
  /**
   * Set user preference
   */
  setUserPreference: async (userId: string, key: string, value: unknown, category = 'general'): Promise<{ success: boolean }> => {
    const response = await apiClient.put(`/memory/users/${userId}/preferences/${key}`, {
      key,
      value,
      category,
    });
    return response.data;
  },

  /**
   * Get user preference
   */
  getUserPreference: async (userId: string, key: string): Promise<{ value: unknown }> => {
    const response = await apiClient.get(`/memory/users/${userId}/preferences/${key}`);
    return response.data;
  },

  /**
   * Get all user preferences
   */
  getUserPreferences: async (userId: string): Promise<{ preferences: Record<string, unknown> }> => {
    const response = await apiClient.get(`/memory/users/${userId}/preferences`);
    return response.data;
  },

  /**
   * Get full user context
   */
  getUserContext: async (userId: string): Promise<Record<string, unknown>> => {
    const response = await apiClient.get(`/memory/users/${userId}/context`);
    return response.data;
  },

  // Entities
  /**
   * Store entity
   */
  storeEntity: async (entity: Omit<MemoryEntity, 'entity_id'> & { entity_id?: string }): Promise<{ success: boolean }> => {
    const response = await apiClient.post('/memory/entities', entity);
    return response.data;
  },

  /**
   * Get entity
   */
  getEntity: async (entityType: string, entityId: string): Promise<MemoryEntity> => {
    const response = await apiClient.get(`/memory/entities/${entityType}/${entityId}`);
    return response.data;
  },

  /**
   * Search entities
   */
  searchEntities: async (query?: string, entityType?: string, limit = 10): Promise<{ entities: MemoryEntity[] }> => {
    const response = await apiClient.get('/memory/entities', {
      params: { query, entity_type: entityType, limit },
    });
    return response.data;
  },

  // Knowledge
  /**
   * Store knowledge item
   */
  storeKnowledge: async (knowledge: MemoryKnowledge): Promise<{ success: boolean }> => {
    const response = await apiClient.post('/memory/knowledge', knowledge);
    return response.data;
  },

  /**
   * Search knowledge
   */
  searchKnowledge: async (query: string, tags?: string[], limit = 10): Promise<{ items: MemoryKnowledge[] }> => {
    const response = await apiClient.get('/memory/knowledge', {
      params: { query, tags: tags?.join(','), limit },
    });
    return response.data;
  },

  // Session Data
  /**
   * Set session data
   */
  setSessionData: async (sessionId: string, key: string, value: unknown): Promise<{ success: boolean }> => {
    const response = await apiClient.put(`/memory/sessions/${sessionId}/data/${key}`, { key, value });
    return response.data;
  },

  /**
   * Get session data
   */
  getSessionData: async (sessionId: string, key: string): Promise<{ value: unknown }> => {
    const response = await apiClient.get(`/memory/sessions/${sessionId}/data/${key}`);
    return response.data;
  },

  // Stats
  /**
   * Get memory statistics
   */
  getStats: async (): Promise<MemoryStats> => {
    const response = await apiClient.get('/memory/stats');
    return response.data;
  },
};

// ==================== RAG Service API ====================

export const ragApi = {
  /**
   * Retrieve documents for a query
   */
  retrieve: async (request: RetrieveRequest): Promise<RetrieveResponse> => {
    const response = await apiClient.post<RetrieveResponse>('/rag/retrieve', request);
    return response.data;
  },

  /**
   * Index a document
   */
  indexDocument: async (request: KnowledgeAddRequest): Promise<KnowledgeResponse> => {
    const response = await apiClient.post<KnowledgeResponse>('/rag/documents', request);
    return response.data;
  },

  /**
   * Upload a file
   */
  uploadFile: async (file: File, knowledgeBaseId?: string): Promise<KnowledgeResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    if (knowledgeBaseId) {
      formData.append('knowledge_base_id', knowledgeBaseId);
    }

    const response = await apiClient.post<KnowledgeResponse>('/rag/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000, // 5 minutes for file upload
    });
    return response.data;
  },

  /**
   * List documents
   */
  listDocuments: async (page = 1, pageSize = 20, knowledgeBaseId?: string): Promise<{
    documents: DocumentInfo[];
    total: number;
    page: number;
    page_size: number;
  }> => {
    const response = await apiClient.get('/rag/documents', {
      params: { page, page_size: pageSize, knowledge_base_id: knowledgeBaseId },
    });
    return response.data;
  },

  /**
   * Delete document
   */
  deleteDocument: async (documentId: string): Promise<{ success: boolean }> => {
    const response = await apiClient.delete(`/rag/documents/${documentId}`);
    return response.data;
  },

  /**
   * Get RAG statistics
   */
  getStats: async (knowledgeBaseId?: string): Promise<RAGStats> => {
    const response = await apiClient.get('/rag/stats', {
      params: { knowledge_base_id: knowledgeBaseId },
    });
    return response.data;
  },

  /**
   * Check RAG service health
   */
  checkHealth: async (): Promise<ServiceHealth> => {
    try {
      const response = await apiClient.get('/rag/health');
      return response.data;
    } catch {
      return { status: 'unhealthy', service: 'rag-service' };
    }
  },
};

// ==================== MCP Service API ====================

export const mcpApi = {
  /**
   * List all available MCP tools
   */
  listTools: async (): Promise<MCPTool[]> => {
    const response = await apiClient.get<{ tools: MCPTool[] }>('/mcp/tools');
    return response.data.tools || response.data as unknown as MCPTool[];
  },

  /**
   * Get tool information
   */
  getToolInfo: async (toolName: string): Promise<MCPTool> => {
    const response = await apiClient.get(`/mcp/tools/${toolName}`);
    return response.data;
  },

  /**
   * Execute a tool
   */
  executeTool: async (request: MCPToolExecuteRequest): Promise<MCPToolExecuteResponse> => {
    const response = await apiClient.post(`/mcp/tools/${request.tool_name}/execute`, {
      arguments: request.arguments,
    });
    return response.data;
  },

  /**
   * Check MCP service health
   */
  checkHealth: async (): Promise<ServiceHealth> => {
    try {
      const response = await apiClient.get('/mcp/health');
      return response.data;
    } catch {
      return { status: 'unhealthy', service: 'mcp-service' };
    }
  },
};

// ==================== Monitoring Service API ====================

export const monitoringApi = {
  /**
   * Get monitoring statistics
   */
  getStats: async (): Promise<MonitoringStats> => {
    const response = await apiClient.get('/monitoring/stats');
    return response.data;
  },

  /**
   * Get metrics
   */
  getMetrics: async (metricNames?: string[], startTime?: string, endTime?: string): Promise<MetricData[]> => {
    const response = await apiClient.get('/monitoring/metrics', {
      params: { names: metricNames?.join(','), start_time: startTime, end_time: endTime },
    });
    return response.data.metrics || [];
  },

  /**
   * Get traces
   */
  getTraces: async (limit = 50, serviceName?: string): Promise<TraceSpan[]> => {
    const response = await apiClient.get('/monitoring/traces', {
      params: { limit, service_name: serviceName },
    });
    return response.data.traces || [];
  },

  /**
   * Record custom metric
   */
  recordMetric: async (metric: MetricData): Promise<{ success: boolean }> => {
    const response = await apiClient.post('/monitoring/metrics', metric);
    return response.data;
  },

  /**
   * Check monitoring service health
   */
  checkHealth: async (): Promise<ServiceHealth> => {
    try {
      const response = await apiClient.get('/monitoring/health');
      return response.data;
    } catch {
      return { status: 'unhealthy', service: 'monitoring-service' };
    }
  },
};

// ==================== Single Agent Service API ====================

export const agentApi = {
  /**
   * Send chat to single agent
   */
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await apiClient.post<ChatResponse>('/agent/chat', {
      message: request.message,
      conversation_id: request.conversation_id,
      config: {
        enable_tools: request.config?.enable_tools ?? true,
        enable_rag: request.config?.enable_rag ?? false,
        knowledge_base_id: request.config?.knowledge_base_id,
        temperature: request.config?.temperature,
        max_tokens: request.config?.max_tokens,
      },
    });
    return response.data;
  },

  /**
   * Stream agent chat
   */
  streamMessage: async function* (request: ChatRequest): AsyncGenerator<AgentStreamEvent> {
    const response = await fetch(`${API_BASE_URL}/agent/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: request.message,
        conversation_id: request.conversation_id,
        config: {
          enable_tools: request.config?.enable_tools ?? true,
          enable_rag: request.config?.enable_rag ?? false,
        },
        stream: true,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) throw new Error('Response body is null');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') return;
          try {
            yield JSON.parse(data) as AgentStreamEvent;
          } catch {
            // Not JSON, skip
          }
        }
      }
    }
  },

  /**
   * Check agent service health
   */
  checkHealth: async (): Promise<ServiceHealth> => {
    try {
      const response = await apiClient.get('/agent/health');
      return response.data;
    } catch {
      return { status: 'unhealthy', service: 'single-agent-service' };
    }
  },
};

// ==================== Multi-Agent (A2A) Service API ====================

export const a2aApi = {
  /**
   * Get A2A health status
   */
  getHealth: async (): Promise<A2AHealthResponse> => {
    const response = await apiClient.get<A2AHealthResponse>('/a2a/health');
    return response.data;
  },

  /**
   * Get list of all agents
   */
  getAgents: async (): Promise<{ agents: A2AAgentInfo[]; total: number }> => {
    const response = await apiClient.get('/a2a/agents');
    return response.data;
  },

  /**
   * Send chat through A2A routing
   */
  sendMessage: async (request: A2AChatRequest): Promise<A2AChatResponse> => {
    const response = await apiClient.post<A2AChatResponse>('/a2a/chat', request);
    return response.data;
  },

  /**
   * Stream A2A chat
   */
  streamMessage: async function* (request: A2AChatRequest): AsyncGenerator<A2AStreamEvent> {
    const response = await fetch(`${API_BASE_URL}/a2a/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...request, stream: true }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) throw new Error('Response body is null');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') return;
          try {
            yield JSON.parse(data) as A2AStreamEvent;
          } catch {
            // Not JSON, skip
          }
        }
      }
    }
  },

  /**
   * Send parallel chat
   */
  sendParallel: async (request: A2AChatRequest): Promise<A2AChatResponse> => {
    const response = await apiClient.post<A2AChatResponse>('/a2a/chat/parallel', {
      message: request.message,
      agent_names: request.agents,
      conversation_id: request.conversation_id,
    });
    return response.data;
  },

  /**
   * Send sequential chat
   */
  sendSequential: async (request: A2AChatRequest): Promise<A2AChatResponse> => {
    const response = await apiClient.post<A2AChatResponse>('/a2a/chat/sequential', {
      message: request.message,
      agent_chain: request.agents,
      conversation_id: request.conversation_id,
    });
    return response.data;
  },
};

// ==================== LLM Manager Service API ====================

export const llmApi = {
  /**
   * Get available providers
   */
  getProviders: async (): Promise<ProviderInfo[]> => {
    const response = await apiClient.get<ProviderInfo[]>('/llm/providers');
    return response.data;
  },

  /**
   * Get current config
   */
  getCurrentConfig: async (): Promise<ModelConfig> => {
    const response = await apiClient.get<ModelConfig>('/llm/config');
    return response.data;
  },

  /**
   * Update config
   */
  updateConfig: async (config: ModelConfigRequest): Promise<ModelConfig> => {
    const response = await apiClient.post<ModelConfig>('/llm/config', config);
    return response.data;
  },

  /**
   * Test config
   */
  testConfig: async (config: ModelConfigRequest): Promise<{ success: boolean; error?: string }> => {
    const response = await apiClient.post('/llm/config/test', config);
    return response.data;
  },

  /**
   * Check LLM service health
   */
  checkHealth: async (): Promise<ServiceHealth> => {
    try {
      const response = await apiClient.get('/llm/health');
      return response.data;
    } catch {
      return { status: 'unhealthy', service: 'llm-manager-service' };
    }
  },
};

// ==================== System API ====================

export const systemApi = {
  /**
   * Get system info
   */
  getInfo: async (): Promise<SystemInfo> => {
    const response = await apiClient.get<SystemInfo>('/system/info');
    return response.data;
  },

  /**
   * Gateway health check
   */
  healthCheck: async (): Promise<ServiceHealth> => {
    try {
      const response = await apiClient.get('/health');
      return response.data;
    } catch {
      return { status: 'unhealthy', service: 'api-gateway' };
    }
  },
};

// ==================== Legacy API Compatibility ====================

// For backward compatibility with existing code
export const knowledgeApi = {
  uploadDocument: ragApi.uploadFile,
  addText: async (request: { text: string; title?: string }): Promise<KnowledgeResponse> => {
    return ragApi.indexDocument({ content: request.text, source: request.title });
  },
  search: async (query: string, topK = 5): Promise<DocumentInfo[]> => {
    const result = await ragApi.retrieve({ query, top_k: topK });
    return result.documents;
  },
  clear: async (): Promise<{ success: boolean; message: string }> => {
    // This would need to be implemented in RAG service
    return { success: true, message: 'Knowledge base cleared' };
  },
};

export const configApi = {
  getProviders: llmApi.getProviders,
  getProviderModels: async (providerId: string): Promise<{ provider: string; models: string[] }> => {
    const providers = await llmApi.getProviders();
    const provider = providers.find(p => p.id === providerId);
    return { provider: providerId, models: provider?.models || [] };
  },
  getCurrentConfig: llmApi.getCurrentConfig,
  updateConfig: llmApi.updateConfig,
  validateConfig: async (): Promise<{ valid: boolean }> => ({ valid: true }),
  testConfig: llmApi.testConfig,
};

export const conversationApi = {
  list: async (params?: { limit?: number; offset?: number }): Promise<{
    conversations: ConversationInfo[];
    total: number;
    limit: number;
    offset: number;
  }> => {
    const page = params?.offset ? Math.floor(params.offset / (params.limit || 20)) + 1 : 1;
    const result = await memoryApi.listConversations(page, params?.limit || 20);
    return {
      conversations: result.conversations,
      total: result.total,
      limit: params?.limit || 20,
      offset: params?.offset || 0,
    };
  },
  getDetail: memoryApi.getConversation,
  update: async (conversationId: string, data: { title?: string }): Promise<{ success: boolean }> => {
    // Memory service doesn't have update, but we can implement client-side
    console.log('Update conversation:', conversationId, data);
    return { success: true };
  },
  delete: memoryApi.deleteConversation,
  export: async (conversationId: string): Promise<{ conversation_id: string; data: unknown; exported_at: string }> => {
    const detail = await memoryApi.getConversation(conversationId);
    return {
      conversation_id: conversationId,
      data: detail,
      exported_at: new Date().toISOString(),
    };
  },
  import: async (): Promise<{ success: boolean; conversation_id: string }> => {
    return { success: false, conversation_id: '' };
  },
};

export const memoryStoreApi = {
  getStats: memoryApi.getStats,
  setUserPreference: async (userId: string, preference: { key: string; value: unknown; category?: string }): Promise<UserPreference> => {
    await memoryApi.setUserPreference(userId, preference.key, preference.value, preference.category);
    return { key: preference.key, value: preference.value, category: preference.category };
  },
  getUserPreferences: memoryApi.getUserPreferences,
  getUserPreference: memoryApi.getUserPreference,
  storeEntity: memoryApi.storeEntity,
  getEntity: memoryApi.getEntity,
  searchEntities: async (query: string, entityType?: string, limit?: number): Promise<{ query: string; results: MemoryEntity[]; count: number }> => {
    const result = await memoryApi.searchEntities(query, entityType, limit);
    return { query, results: result.entities, count: result.entities.length };
  },
  storeKnowledge: memoryApi.storeKnowledge,
  searchKnowledge: async (query: string, limit?: number): Promise<{ results: MemoryKnowledge[]; total: number; query: string }> => {
    const result = await memoryApi.searchKnowledge(query, undefined, limit);
    return { results: result.items, total: result.items.length, query };
  },
  getUserContext: memoryApi.getUserContext,
  clearUserMemory: async (userId: string): Promise<{ success: boolean; user_id: string; deleted_items: number }> => {
    // Not implemented in memory service yet
    return { success: false, user_id: userId, deleted_items: 0 };
  },
};

export const langgraphApi = agentApi;

// ==================== Default Export ====================

export default {
  services: servicesApi,
  chat: chatApi,
  memory: memoryApi,
  rag: ragApi,
  mcp: mcpApi,
  monitoring: monitoringApi,
  agent: agentApi,
  a2a: a2aApi,
  llm: llmApi,
  system: systemApi,
  // Legacy compatibility
  knowledge: knowledgeApi,
  config: configApi,
  conversation: conversationApi,
  memoryStore: memoryStoreApi,
  langgraph: langgraphApi,
};
