"""
LlamaIndex Index Manager.
Manages the creation, loading, and updating of LlamaIndex indices.

Features:
- Document deduplication (skip already embedded documents)
- Async batch embedding for better performance
- Text preprocessing for improved LLM understanding
- Automatic duplicate cleanup
"""
import hashlib
import logging
import re
import time
from typing import List, Optional, Dict, Any, Set, Tuple
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings as LlamaSettings,
)
from llama_index.core.schema import Document as LlamaDocument, TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama

from config.settings import settings
from app.core.embeddings import get_embedding_manager

logger = logging.getLogger(__name__)


def compute_content_hash(text: str) -> str:
    """Compute MD5 hash of text content for deduplication."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def preprocess_text_for_embedding(text: str) -> str:
    """
    Preprocess text to improve embedding quality and LLM understanding.

    Improvements:
    - Normalize whitespace
    - Remove excessive punctuation
    - Clean special characters
    - Preserve semantic structure
    """
    if not text:
        return text

    # 1. Normalize whitespace (preserve paragraph breaks)
    text = re.sub(r'[ \t]+', ' ', text)  # Collapse multiple spaces/tabs
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines

    # 2. Remove control characters except newlines
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 3. Normalize common encoding issues
    replacements = {
        '\u200b': '',  # Zero-width space
        '\u200c': '',  # Zero-width non-joiner
        '\u200d': '',  # Zero-width joiner
        '\ufeff': '',  # BOM
        '…': '...',
        '—': '-',
        '–': '-',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 4. Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # 5. Final cleanup
    text = text.strip()

    return text


class IndexManager:
    """
    Manages LlamaIndex vector store indices with Milvus backend.

    Features:
    - Document deduplication using content hash
    - Async batch embedding for performance
    - Text preprocessing for better LLM understanding
    - Automatic duplicate detection and cleanup
    - Reuses global EmbeddingManager for consistency
    """

    # Class-level flag to track if settings have been configured
    _settings_configured: bool = False

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        self.collection_name = collection_name or settings.MILVUS_COLLECTION
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.llm_model = llm_model or settings.LLM_MODEL

        self._vector_store: Optional[MilvusVectorStore] = None
        self._index: Optional[VectorStoreIndex] = None
        self._embedding_dim: Optional[int] = None

        # Track indexed content hashes for deduplication
        # Maps content_hash -> node_id
        self._indexed_hashes: Dict[str, str] = {}

        # Configure LlamaIndex global settings (only once)
        self._configure_settings()

        logger.info(
            f"[IndexManager] Initialized - collection: {self.collection_name}, "
            f"embedding: {self.embedding_model}, llm: {self.llm_model}"
        )

    def _configure_settings(self):
        """
        Configure LlamaIndex global settings.

        Uses the global EmbeddingManager's LlamaIndex instance to ensure
        consistency and reuse the warmed-up embedding model.
        """
        # Only configure once per process
        if IndexManager._settings_configured:
            return

        logger.info("[IndexManager] Configuring LlamaIndex global settings")

        # Reuse global EmbeddingManager's LlamaIndex embedding instance
        # This ensures we use the same warmed-up model
        embedding_manager = get_embedding_manager()
        LlamaSettings.embed_model = embedding_manager.llamaindex

        # Set LLM
        LlamaSettings.llm = Ollama(
            model=self.llm_model,
            base_url=settings.LLM_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            request_timeout=120.0,
        )

        # Set chunk sizes
        LlamaSettings.chunk_size = settings.RAG_CHUNK_SIZE
        LlamaSettings.chunk_overlap = settings.RAG_CHUNK_OVERLAP

        IndexManager._settings_configured = True
        logger.info("[IndexManager] LlamaIndex settings configured (using global EmbeddingManager)")

    @property
    def vector_store(self) -> MilvusVectorStore:
        """Get or create the Milvus vector store."""
        if self._vector_store is None:
            logger.info(f"[IndexManager] Creating MilvusVectorStore - {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
            self._vector_store = MilvusVectorStore(
                uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
                collection_name=self.collection_name,
                dim=self._get_embedding_dimension(),
                overwrite=False,  # Don't overwrite existing collection
                embedding_field="vector",  # Match existing Milvus schema
                text_key="text",  # Match existing Milvus schema
                enable_dynamic_field=True,  # 支持动态 metadata 字段
            )
        return self._vector_store

    @property
    def index(self) -> VectorStoreIndex:
        """Get or create the vector store index."""
        if self._index is None:
            logger.info("[IndexManager] Creating VectorStoreIndex from existing store")
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=storage_context,
            )
        return self._index

    def _get_embedding_dimension(self) -> int:
        """
        Get embedding dimension.

        Uses cached dimension from EmbeddingManager if available,
        otherwise computes by embedding a test string.
        """
        if self._embedding_dim is not None:
            return self._embedding_dim

        # Try to get from EmbeddingManager first (may be cached from warmup)
        try:
            embedding_manager = get_embedding_manager()
            if embedding_manager._embedding_dim is not None:
                self._embedding_dim = embedding_manager._embedding_dim
                return self._embedding_dim
        except Exception:
            pass

        # Fallback: compute by embedding test string
        embed_model = LlamaSettings.embed_model
        test_embedding = embed_model.get_text_embedding("test")
        self._embedding_dim = len(test_embedding)
        return self._embedding_dim

    def _compute_node_hash(self, node: TextNode) -> str:
        """
        Compute hash for a node based on content and source.

        Uses combination of:
        - Content text
        - Source file (if available)
        """
        content = node.get_content()
        source = node.metadata.get("source", "") or node.metadata.get("file_path", "")
        combined = f"{source}:{content}"
        return compute_content_hash(combined)

    def _check_duplicates(
        self,
        nodes: List[TextNode],
    ) -> Tuple[List[TextNode], List[TextNode], Dict[str, str]]:
        """
        Check for duplicate nodes.

        Args:
            nodes: List of nodes to check

        Returns:
            Tuple of (new_nodes, duplicate_nodes, hash_to_node_id_map)
        """
        new_nodes = []
        duplicate_nodes = []
        hash_to_node_id = {}

        for node in nodes:
            content_hash = self._compute_node_hash(node)

            if content_hash in self._indexed_hashes:
                # Already indexed
                duplicate_nodes.append(node)
                logger.debug(
                    f"[IndexManager] Duplicate found: {node.metadata.get('source', 'unknown')[:50]}"
                )
            else:
                new_nodes.append(node)
                hash_to_node_id[content_hash] = node.node_id

        return new_nodes, duplicate_nodes, hash_to_node_id

    def _delete_nodes_by_ids(self, node_ids: List[str]) -> int:
        """
        Delete nodes from Milvus by their IDs.

        Args:
            node_ids: List of node IDs to delete

        Returns:
            Number of deleted nodes
        """
        if not node_ids:
            return 0

        try:
            from pymilvus import Collection, connections

            # Connect to Milvus
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
            )

            collection = Collection(self.collection_name)
            collection.load()

            # Delete by IDs - Milvus uses 'id' as primary key field
            # The node_id is stored in the 'id' field
            expr = f"id in {node_ids}"
            result = collection.delete(expr)

            deleted_count = result.delete_count if hasattr(result, 'delete_count') else len(node_ids)
            logger.info(f"[IndexManager] Deleted {deleted_count} duplicate nodes from Milvus")

            return deleted_count

        except Exception as e:
            logger.warning(f"[IndexManager] Failed to delete nodes: {e}")
            return 0

    def _delete_duplicates_by_source(self, source: str) -> int:
        """
        Delete all nodes with a specific source file.

        Args:
            source: Source file path

        Returns:
            Number of deleted nodes
        """
        if not source:
            return 0

        try:
            from pymilvus import Collection, connections

            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
            )

            collection = Collection(self.collection_name)
            collection.load()

            # Delete by source metadata
            # Note: This requires 'source' field to exist in schema
            expr = f'source == "{source}"'
            result = collection.delete(expr)

            deleted_count = result.delete_count if hasattr(result, 'delete_count') else 0
            if deleted_count > 0:
                logger.info(f"[IndexManager] Deleted {deleted_count} nodes for source: {source}")

            # Remove from hash cache
            hashes_to_remove = [
                h for h, nid in self._indexed_hashes.items()
                if nid.startswith(source)
            ]
            for h in hashes_to_remove:
                del self._indexed_hashes[h]

            return deleted_count

        except Exception as e:
            logger.warning(f"[IndexManager] Failed to delete by source: {e}")
            return 0

    def add_documents(
        self,
        documents: List[LlamaDocument],
        show_progress: bool = True,
    ) -> List[str]:
        """
        Add documents to the index.

        Args:
            documents: List of LlamaIndex documents
            show_progress: Whether to show progress bar

        Returns:
            List of node IDs
        """
        logger.info(f"[IndexManager] Adding {len(documents)} documents to index")

        # Create or update index
        if self._index is None:
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self._index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=show_progress,
            )
            logger.info("[IndexManager] Created new index with documents")
        else:
            # Insert into existing index
            for doc in documents:
                self._index.insert(doc)
            logger.info("[IndexManager] Inserted documents into existing index")

        return [doc.doc_id for doc in documents]

    def add_nodes(
        self,
        nodes: List[TextNode],
        show_progress: bool = True,
    ) -> List[str]:
        """
        Add pre-processed nodes to the index.

        Args:
            nodes: List of TextNode objects
            show_progress: Whether to show progress bar

        Returns:
            List of node IDs
        """
        logger.info(f"[IndexManager] Adding {len(nodes)} nodes to index")

        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        if self._index is None:
            self._index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                show_progress=show_progress,
            )
        else:
            self._index.insert_nodes(nodes)

        return [node.node_id for node in nodes]

    async def add_nodes_async(
        self,
        nodes: List[TextNode],
        batch_size: Optional[int] = None,
        max_concurrent: Optional[int] = None,
        show_progress: bool = True,
        skip_duplicates: bool = True,
        delete_existing_source: bool = True,
        preprocess_text: bool = True,
    ) -> List[str]:
        """
        Async add pre-processed nodes with batch embedding optimization.

        This method pre-computes embeddings using batch async processing
        before adding to the vector store, significantly improving performance.

        Features:
        - Text preprocessing for better LLM understanding
        - Deduplication (skip already embedded content)
        - Auto-delete existing nodes when re-uploading same source file
        - Batch async embedding for performance

        Args:
            nodes: List of TextNode objects
            batch_size: Number of documents per embedding batch
            max_concurrent: Maximum concurrent embedding requests
            show_progress: Whether to log progress
            skip_duplicates: Skip nodes that are already indexed (by content hash)
            delete_existing_source: Delete existing nodes with same source before adding
            preprocess_text: Apply text preprocessing for better embedding quality

        Returns:
            List of node IDs
        """
        if not nodes:
            return []

        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        max_concurrent = max_concurrent or settings.EMBEDDING_MAX_CONCURRENT

        original_count = len(nodes)
        total_chars = sum(len(n.get_content()) for n in nodes)
        logger.info(f"[IndexManager] ========== Async Batch Indexing Start ==========")
        logger.info(f"[IndexManager] Total nodes: {len(nodes)}")
        logger.info(f"[IndexManager] Total chars: {total_chars:,}")
        logger.info(f"[IndexManager] Batch size: {batch_size}")
        logger.info(f"[IndexManager] Max concurrent: {max_concurrent}")
        logger.info(f"[IndexManager] Skip duplicates: {skip_duplicates}")
        logger.info(f"[IndexManager] Preprocess text: {preprocess_text}")

        start_time = time.time()

        # Step 0: Preprocess text for better embedding quality
        if preprocess_text:
            logger.info(f"[IndexManager] 🔧 Preprocessing text for better LLM understanding...")
            for node in nodes:
                original_text = node.get_content()
                processed_text = preprocess_text_for_embedding(original_text)
                if processed_text != original_text:
                    node.set_content(processed_text)
            logger.info(f"[IndexManager] ✅ Text preprocessing complete")

        # Step 1: Delete existing nodes with same source (for re-upload scenario)
        if delete_existing_source:
            sources_to_delete: Set[str] = set()
            for node in nodes:
                source = node.metadata.get("source") or node.metadata.get("file_path")
                if source:
                    sources_to_delete.add(source)

            if sources_to_delete:
                logger.info(f"[IndexManager] 🗑️ Checking for existing sources to delete...")
                total_deleted = 0
                for source in sources_to_delete:
                    deleted = self._delete_duplicates_by_source(source)
                    total_deleted += deleted
                if total_deleted > 0:
                    logger.info(f"[IndexManager] ✅ Deleted {total_deleted} existing nodes for re-indexing")

        # Step 2: Check for duplicate content (skip already indexed)
        nodes_to_embed = nodes
        skipped_count = 0

        if skip_duplicates:
            logger.info(f"[IndexManager] 🔍 Checking for duplicate content...")
            new_nodes, duplicate_nodes, hash_to_node_id = self._check_duplicates(nodes)
            skipped_count = len(duplicate_nodes)

            if skipped_count > 0:
                logger.info(f"[IndexManager] ⏭️ Skipping {skipped_count} duplicate nodes (already indexed)")
                nodes_to_embed = new_nodes

        if not nodes_to_embed:
            logger.info(f"[IndexManager] ✅ All nodes already indexed, nothing to do")
            return [node.node_id for node in nodes]

        # Step 3: Extract texts from nodes
        texts = [node.get_content() for node in nodes_to_embed]

        # Step 4: Batch async embed all texts
        logger.info(f"[IndexManager] 🚀 Generating embeddings for {len(nodes_to_embed)} nodes...")
        embedding_manager = get_embedding_manager()
        embeddings = await embedding_manager.aembed_documents_batch(
            texts,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            show_progress=show_progress,
        )

        # Step 5: Set embeddings on nodes
        logger.info(f"[IndexManager] 📝 Setting embeddings on nodes...")
        for node, embedding in zip(nodes_to_embed, embeddings):
            node.embedding = embedding

        # Step 6: Add nodes to vector store (embeddings already computed)
        logger.info(f"[IndexManager] 💾 Adding nodes to vector store...")
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        if self._index is None:
            # When nodes have embeddings, VectorStoreIndex skips embedding step
            self._index = VectorStoreIndex(
                nodes=nodes_to_embed,
                storage_context=storage_context,
                show_progress=show_progress,
            )
        else:
            self._index.insert_nodes(nodes_to_embed)

        # Step 7: Update hash cache for deduplication
        if skip_duplicates:
            for node in nodes_to_embed:
                content_hash = self._compute_node_hash(node)
                self._indexed_hashes[content_hash] = node.node_id

        elapsed = time.time() - start_time
        speed = len(nodes_to_embed) / elapsed if elapsed > 0 else 0

        logger.info(f"[IndexManager] ========== Async Batch Indexing Complete ==========")
        logger.info(
            f"[IndexManager] ✅ Indexed {len(nodes_to_embed)}/{original_count} nodes "
            f"(skipped {skipped_count} duplicates)"
        )
        logger.info(
            f"[IndexManager] ⏱️ Time: {elapsed:.2f}s | Speed: {speed:.1f} nodes/sec"
        )

        return [node.node_id for node in nodes_to_embed]

    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        Delete documents from the index.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            True if successful
        """
        logger.info(f"[IndexManager] Deleting {len(doc_ids)} documents from index")

        if self._index is not None:
            for doc_id in doc_ids:
                self._index.delete_ref_doc(doc_id)

        return True

    def get_retriever(
        self,
        similarity_top_k: int = None,
        **kwargs,
    ):
        """
        Get a retriever from the index.

        Args:
            similarity_top_k: Number of top similar documents to retrieve
            **kwargs: Additional arguments for the retriever

        Returns:
            VectorStoreRetriever
        """
        top_k = similarity_top_k or settings.RAG_RETRIEVAL_TOP_K
        logger.info(f"[IndexManager] Creating retriever with top_k={top_k}")

        return self.index.as_retriever(
            similarity_top_k=top_k,
            **kwargs,
        )

    def get_query_engine(self, **kwargs):
        """
        Get a query engine from the index.

        Args:
            **kwargs: Additional arguments for the query engine

        Returns:
            QueryEngine
        """
        logger.info("[IndexManager] Creating query engine")
        return self.index.as_query_engine(**kwargs)

    def clear_index(self) -> bool:
        """Clear all documents from the index."""
        logger.warning("[IndexManager] Clearing all documents from index")

        # Clear the content hash cache to allow re-indexing
        self._indexed_hashes.clear()
        logger.info("[IndexManager] 🗑️ Cleared content hash cache")

        # Recreate vector store with overwrite=True
        self._vector_store = MilvusVectorStore(
            uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            collection_name=self.collection_name,
            dim=self._get_embedding_dimension(),
            overwrite=True,
            embedding_field="vector",  # Match existing Milvus schema
            text_key="text",  # Match existing Milvus schema
            enable_dynamic_field=True,  # 支持动态 metadata 字段
        )
        self._index = None

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        stats = {
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
        }

        try:
            if self._vector_store is not None:
                stats["is_connected"] = True
            else:
                stats["is_connected"] = False
        except Exception as e:
            stats["is_connected"] = False
            stats["error"] = str(e)

        return stats

    async def get_all_nodes_async(self, limit: int = 10000) -> List[TextNode]:
        """
        从 Milvus 获取所有已索引的节点（用于恢复 BM25 索引）。

        注意：这是一个相对昂贵的操作，仅应在服务启动时调用。

        Args:
            limit: 最大获取数量

        Returns:
            List[TextNode]: 节点列表
        """
        logger.info(f"[IndexManager] Fetching all nodes from Milvus (limit={limit})...")

        try:
            from pymilvus import Collection, connections

            # 连接 Milvus
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
            )

            collection = Collection(self.collection_name)

            # 检查集合是否存在和有数据
            try:
                collection.load()
                num_entities = collection.num_entities
                logger.info(f"[IndexManager] Collection has {num_entities} entities")

                if num_entities == 0:
                    return []
            except Exception as e:
                logger.warning(f"[IndexManager] Failed to load collection: {e}")
                return []

            # 查询所有文档
            # 注意：Milvus 需要使用 query 而不是直接遍历
            try:
                # 获取集合的 schema 来确定字段名
                schema = collection.schema
                field_names = [field.name for field in schema.fields]

                # 查找文本字段
                text_field = "text" if "text" in field_names else None
                id_field = "id" if "id" in field_names else "pk"

                # 输出字段列表（不包括向量字段，因为太大）
                output_fields = [f for f in field_names if f not in ["vector", "embedding"]]

                # 使用空查询获取所有数据
                results = collection.query(
                    expr=f"{id_field} != ''",  # 获取所有非空 ID 的记录
                    output_fields=output_fields,
                    limit=limit,
                )

                logger.info(f"[IndexManager] Retrieved {len(results)} records from Milvus")

                # 转换为 TextNode
                nodes = []
                for record in results:
                    text = record.get("text", "")
                    if not text:
                        continue

                    # 构建 metadata
                    metadata = {}
                    for key, value in record.items():
                        if key not in ["text", "vector", "embedding", "id", "pk"]:
                            metadata[key] = value

                    node = TextNode(
                        text=text,
                        id_=record.get("id") or record.get("pk"),
                        metadata=metadata,
                    )
                    nodes.append(node)

                logger.info(f"[IndexManager] ✅ Converted {len(nodes)} records to TextNodes")
                return nodes

            except Exception as e:
                logger.warning(f"[IndexManager] Failed to query collection: {e}")
                return []

        except Exception as e:
            logger.error(f"[IndexManager] Error fetching all nodes: {e}", exc_info=True)
            return []


# Global singleton
_index_manager: Optional[IndexManager] = None


def get_index_manager() -> IndexManager:
    """Get the global index manager instance."""
    global _index_manager
    if _index_manager is None:
        _index_manager = IndexManager()
    return _index_manager
