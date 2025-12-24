"""
Knowledge Base Service - Auto-indexing and management.

Provides automatic knowledge base indexing on startup and query fallback logic.

功能增强 (v2.0):
- 支持 BM25 索引恢复，确保服务重启后混合检索正常工作
- 添加检索诊断功能
- 优化初始化流程，确保混合检索组件同步初始化
"""
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from config.settings import settings

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """
    Knowledge Base Service for auto-indexing and management.

    Features:
    - Auto-index documents from knowledge base directory on startup
    - Check if documents are already indexed
    - Provide query fallback logic
    """

    def __init__(self):
        self.knowledge_base_path = Path(settings.KNOWLEDGE_BASE_PATH)
        self._indexed_files: set = set()
        self._is_initialized = False
        logger.info(f"[KnowledgeBase] Service created - path: {self.knowledge_base_path}")

    def get_knowledge_base_path(self) -> Path:
        """Get the absolute path to knowledge base directory."""
        # Convert to absolute path if relative
        if not self.knowledge_base_path.is_absolute():
            # Relative to backend directory
            backend_dir = Path(__file__).parent.parent.parent
            return backend_dir / self.knowledge_base_path
        return self.knowledge_base_path

    def list_knowledge_files(self) -> List[Dict[str, Any]]:
        """List all files in the knowledge base directory."""
        kb_path = self.get_knowledge_base_path()

        if not kb_path.exists():
            logger.warning(f"[KnowledgeBase] Directory not found: {kb_path}")
            return []

        files = []
        supported_extensions = {
            '.txt', '.md', '.pdf', '.docx', '.doc',
            '.xlsx', '.xls', '.html', '.htm', '.csv', '.json'
        }

        for file_path in kb_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'extension': file_path.suffix.lower(),
                    'is_indexed': str(file_path) in self._indexed_files,
                })

        return files

    async def initialize(self, rag_service=None) -> Dict[str, Any]:
        """
        Initialize knowledge base by indexing all documents.

        Args:
            rag_service: RAGService instance for indexing documents

        Returns:
            Dict with initialization status and statistics
        """
        if self._is_initialized:
            logger.info("[KnowledgeBase] Already initialized, checking BM25 index")
            # 即使已初始化，也检查并恢复 BM25 索引
            bm25_status = await self._rebuild_bm25_index()
            return {
                'success': True,
                'status': 'already_initialized',
                'indexed_files': len(self._indexed_files),
                'bm25_status': bm25_status
            }

        kb_path = self.get_knowledge_base_path()
        logger.info(f"[KnowledgeBase] Initializing from: {kb_path}")

        # Check if directory exists
        if not kb_path.exists():
            logger.warning(f"[KnowledgeBase] Creating directory: {kb_path}")
            kb_path.mkdir(parents=True, exist_ok=True)
            return {
                'success': True,
                'status': 'directory_created',
                'message': f'Knowledge base directory created at {kb_path}',
                'indexed_files': 0
            }

        # List files to index
        files = self.list_knowledge_files()
        if not files:
            logger.info("[KnowledgeBase] No files found to index")
            self._is_initialized = True
            return {
                'success': True,
                'status': 'no_files',
                'message': 'No documents found in knowledge base',
                'indexed_files': 0
            }

        logger.info(f"[KnowledgeBase] Found {len(files)} files to index")

        # Index documents
        if rag_service is None:
            logger.warning("[KnowledgeBase] RAG service not provided, skipping indexing")
            return {
                'success': False,
                'status': 'no_rag_service',
                'message': 'RAG service not available',
                'files_found': len(files)
            }

        try:
            result = await rag_service.async_index_directory(str(kb_path))

            if result.get('success'):
                # Track indexed files
                for file_info in files:
                    self._indexed_files.add(file_info['path'])

                self._is_initialized = True
                logger.info(
                    f"[KnowledgeBase] Indexing complete - "
                    f"documents: {result.get('num_documents', 0)}, "
                    f"nodes: {result.get('num_nodes', 0)}"
                )

                # 索引完成后，重建 BM25 索引以确保混合检索正常工作
                bm25_status = await self._rebuild_bm25_index()

                return {
                    'success': True,
                    'status': 'indexed',
                    'num_documents': result.get('num_documents', 0),
                    'num_nodes': result.get('num_nodes', 0),
                    'files_indexed': len(files),
                    'files': [f['name'] for f in files],
                    'bm25_status': bm25_status
                }
            else:
                logger.error(f"[KnowledgeBase] Indexing failed: {result.get('error')}")
                return {
                    'success': False,
                    'status': 'index_failed',
                    'error': result.get('error'),
                    'files_found': len(files)
                }

        except Exception as e:
            logger.error(f"[KnowledgeBase] Indexing error: {e}", exc_info=True)
            return {
                'success': False,
                'status': 'error',
                'error': str(e),
                'files_found': len(files)
            }

    def is_initialized(self) -> bool:
        """Check if knowledge base has been initialized."""
        return self._is_initialized

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        kb_path = self.get_knowledge_base_path()
        files = self.list_knowledge_files()

        total_size = sum(f['size'] for f in files)

        return {
            'path': str(kb_path),
            'exists': kb_path.exists(),
            'is_initialized': self._is_initialized,
            'total_files': len(files),
            'indexed_files': len(self._indexed_files),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'files': files
        }

    async def _rebuild_bm25_index(self) -> Dict[str, Any]:
        """
        重建 BM25 索引以确保混合检索正常工作。

        Returns:
            Dict with BM25 rebuild status
        """
        try:
            from app.rag.hybrid_retriever import get_hybrid_retriever

            hybrid_retriever = get_hybrid_retriever()

            # 检查 BM25 是否已经就绪
            if hybrid_retriever.is_bm25_ready():
                stats = hybrid_retriever.get_bm25_stats()
                logger.info(f"[KnowledgeBase] BM25 index already ready: {stats}")
                return {
                    'success': True,
                    'status': 'already_ready',
                    'stats': stats
                }

            # 从向量存储重建 BM25 索引
            logger.info("[KnowledgeBase] Rebuilding BM25 index from vector store...")
            success = await hybrid_retriever.rebuild_bm25_from_vector_store()

            if success:
                stats = hybrid_retriever.get_bm25_stats()
                logger.info(f"[KnowledgeBase] BM25 index rebuilt successfully: {stats}")
                return {
                    'success': True,
                    'status': 'rebuilt',
                    'stats': stats
                }
            else:
                logger.warning("[KnowledgeBase] Failed to rebuild BM25 index")
                return {
                    'success': False,
                    'status': 'rebuild_failed',
                    'message': 'Could not rebuild BM25 index from vector store'
                }

        except ImportError as e:
            logger.warning(f"[KnowledgeBase] HybridRetriever not available: {e}")
            return {
                'success': False,
                'status': 'not_available',
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"[KnowledgeBase] Error rebuilding BM25 index: {e}", exc_info=True)
            return {
                'success': False,
                'status': 'error',
                'error': str(e)
            }

    async def get_retrieval_diagnostics(self, test_query: str = None) -> Dict[str, Any]:
        """
        获取检索系统诊断信息。

        Args:
            test_query: 可选的测试查询，用于验证检索功能

        Returns:
            Dict with diagnostic information
        """
        diagnostics = {
            'knowledge_base': self.get_stats(),
            'components': {},
            'test_results': None
        }

        # 检查 BM25 状态
        try:
            from app.rag.hybrid_retriever import get_hybrid_retriever
            hybrid_retriever = get_hybrid_retriever()
            diagnostics['components']['bm25'] = {
                'ready': hybrid_retriever.is_bm25_ready(),
                'stats': hybrid_retriever.get_bm25_stats()
            }
        except Exception as e:
            diagnostics['components']['bm25'] = {
                'ready': False,
                'error': str(e)
            }

        # 检查向量存储状态
        try:
            from app.rag.index_manager import get_index_manager
            index_manager = get_index_manager()
            diagnostics['components']['vector_store'] = {
                'initialized': index_manager._index is not None,
                'collection': getattr(index_manager, '_collection_name', 'unknown')
            }
        except Exception as e:
            diagnostics['components']['vector_store'] = {
                'initialized': False,
                'error': str(e)
            }

        # 检查重排序器状态
        try:
            from app.rag.reranker import get_jina_reranker, CROSS_ENCODER_AVAILABLE
            diagnostics['components']['reranker'] = {
                'cross_encoder_available': CROSS_ENCODER_AVAILABLE,
                'enabled': True
            }
        except Exception as e:
            diagnostics['components']['reranker'] = {
                'available': False,
                'error': str(e)
            }

        # 执行测试查询
        if test_query:
            try:
                from app.rag.hybrid_retriever import get_hybrid_retriever
                from llama_index.core.schema import QueryBundle
                hybrid_retriever = get_hybrid_retriever()

                # 执行混合检索
                query_bundle = QueryBundle(query_str=test_query)
                results = await hybrid_retriever.aretrieve(query_bundle)

                diagnostics['test_results'] = {
                    'query': test_query,
                    'num_results': len(results),
                    'results': [
                        {
                            'score': r.score,
                            'text_preview': r.node.get_content()[:100] + '...' if len(r.node.get_content()) > 100 else r.node.get_content(),
                            'metadata': r.node.metadata
                        }
                        for r in results[:3]  # 只返回前3个结果的预览
                    ]
                }

                logger.info(
                    f"[KnowledgeBase] Diagnostic query '{test_query[:30]}...' "
                    f"returned {len(results)} results"
                )

            except Exception as e:
                diagnostics['test_results'] = {
                    'query': test_query,
                    'error': str(e)
                }
                logger.error(f"[KnowledgeBase] Diagnostic query failed: {e}")

        return diagnostics


# Global singleton
_knowledge_base_service: Optional[KnowledgeBaseService] = None


def get_knowledge_base_service() -> KnowledgeBaseService:
    """Get the global KnowledgeBaseService instance."""
    global _knowledge_base_service
    if _knowledge_base_service is None:
        _knowledge_base_service = KnowledgeBaseService()
    return _knowledge_base_service


async def initialize_knowledge_base(rag_service=None) -> Dict[str, Any]:
    """
    Initialize the knowledge base (convenience function).

    This function is called on application startup to index all documents
    in the knowledge base directory.

    Args:
        rag_service: RAGService instance

    Returns:
        Initialization result dict
    """
    service = get_knowledge_base_service()
    return await service.initialize(rag_service)
