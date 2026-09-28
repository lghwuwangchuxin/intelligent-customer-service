import { request, upload } from '../lib/http'
import type {
  DocumentInfo,
  IngestResult,
  KnowledgeStats,
  RetrieveResult,
} from './types'

export const knowledgeApi = {
  retrieve: (query: string, kb_id = 'default', top_n?: number) =>
    request<RetrieveResult>('/knowledge/retrieve', {
      method: 'POST',
      body: { query, kb_id, top_n },
    }),

  ingest: (doc_id: string, text: string, title = '', kb_id = 'default') =>
    request<IngestResult>('/knowledge/documents', {
      method: 'POST',
      body: { doc_id, text, title, kb_id },
    }),

  upload: (file: File, kb_id = 'default') =>
    upload<IngestResult>('/knowledge/documents/upload', file, { kb_id }),

  documents: (kb_id?: string) =>
    request<{ documents: DocumentInfo[] }>('/knowledge/documents', {
      query: { kb_id },
    }).then((r) => r.documents),

  remove: (doc_id: string) =>
    request<{ deleted_chunks: number }>(`/knowledge/documents/${doc_id}`, { method: 'DELETE' }),

  stats: () => request<KnowledgeStats>('/knowledge/stats'),
}
