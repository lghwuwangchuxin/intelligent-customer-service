"""
Vector Store Manager / 向量存储管理器
=====================================

基于 LlamaIndex 的 Milvus 向量存储实现。

核心功能
--------
1. **文档存储**: 将文档嵌入向量存入 Milvus
2. **相似度检索**: 基于向量的语义搜索
3. **混合检索**: 支持与 BM25 结合使用
4. **双框架兼容**: 同时支持 LlamaIndex 和 LangChain
5. **Langfuse 追踪**: 自动记录嵌入和检索操作

架构说明
--------
```
LlamaIndex TextNode/Document
           ↓
    OllamaEmbedding (嵌入) → Langfuse Span
           ↓
    MilvusVectorStore (存储)
           ↓
    VectorStoreIndex (索引)
           ↓
    VectorIndexRetriever (检索) → Langfuse Span
```

Langfuse 追踪
-------------
```
Trace: vector_store_operation
├── Span: embedding
│   ├── model
│   ├── num_texts
│   └── elapsed_time
└── Span: retrieval
    ├── query
    ├── top_k
    ├── num_results
    └── scores
```

配置参数
--------
```python
# config/settings.py
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_COLLECTION = "customer_service_kb"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_BASE_URL = "http://localhost:11434"
LANGFUSE_ENABLED = True
```

使用示例
--------
```python
from app.core.vector_store import VectorStoreManager

# 初始化
manager = VectorStoreManager()

# 添加 LlamaIndex 节点
manager.add_nodes(nodes)

# 添加 LangChain 文档 (自动转换)
manager.add_documents(langchain_docs)

# 检索
results = manager.retrieve("查询问题", top_k=5)

# 获取 LlamaIndex 索引 (用于 RAG 管道)
index = manager.get_index()
```

Author: Intelligent Customer Service Team
Version: 2.1.0 (LlamaIndex + Langfuse)
"""
import logging
import time
from typing import List, Dict, Any, Optional, Union

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings as LlamaSettings,
)
from llama_index.core.schema import TextNode, NodeWithScore, BaseNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

# LangChain 兼容
from langchain_core.documents import Document as LangchainDocument
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from config.settings import settings

# Langfuse observability
from app.services.langfuse_service import get_langfuse_service

logger = logging.getLogger(__name__)


class VectorStoreRetriever(BaseRetriever):
    """
    LangChain BaseRetriever 实现。

    继承自 BaseRetriever 以确保与 LCEL 链兼容。

    Attributes
    ----------
    manager : Any
        VectorStoreManager 实例引用
    top_k : int
        返回结果数量
    """

    manager: Any  # VectorStoreManager instance
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[LangchainDocument]:
        """
        获取相关文档。

        Parameters
        ----------
        query : str
            查询文本
        run_manager : CallbackManagerForRetrieverRun, optional
            回调管理器

        Returns
        -------
        List[LangchainDocument]
            相关文档列表
        """
        return self.manager.similarity_search(query, k=self.top_k)


class VectorStoreManager:
    """
    向量存储管理器 (LlamaIndex + Milvus)
    ===================================

    管理 Milvus 向量数据库的连接、索引和检索操作。

    Attributes
    ----------
    host : str
        Milvus 服务器地址

    port : int
        Milvus 端口

    collection_name : str
        集合名称

    embed_model : OllamaEmbedding
        嵌入模型实例

    vector_store : MilvusVectorStore
        Milvus 向量存储实例

    Example
    -------
    ```python
    manager = VectorStoreManager(
        host="localhost",
        port=19530,
        collection_name="my_kb"
    )

    # 添加文档
    manager.add_nodes(nodes)

    # 检索
    results = manager.retrieve("问题", top_k=5)
    for node in results:
        print(f"Score: {node.score}, Text: {node.text[:100]}")
    ```
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = None,
        embedding_model: str = None,
        embedding_base_url: str = None,
        database: str = None,
        dim: int = 768,
    ):
        """
        初始化向量存储管理器。

        Parameters
        ----------
        host : str, optional
            Milvus 地址，默认使用 settings.MILVUS_HOST

        port : int, optional
            Milvus 端口，默认使用 settings.MILVUS_PORT

        collection_name : str, optional
            集合名称，默认使用 settings.MILVUS_COLLECTION

        embedding_model : str, optional
            嵌入模型名称，默认使用 settings.EMBEDDING_MODEL

        embedding_base_url : str, optional
            Ollama 地址，默认使用 settings.EMBEDDING_BASE_URL

        database : str, optional
            数据库名称

        dim : int, optional
            向量维度，默认 768
        """
        self.host = host or settings.MILVUS_HOST
        self.port = port or settings.MILVUS_PORT
        self.collection_name = collection_name or settings.MILVUS_COLLECTION
        self.database = database or settings.MILVUS_DATABASE
        self.dim = dim

        # 初始化嵌入模型
        self.embed_model = OllamaEmbedding(
            model_name=embedding_model or settings.EMBEDDING_MODEL,
            base_url=embedding_base_url or settings.EMBEDDING_BASE_URL,
        )

        # 设置全局嵌入模型
        LlamaSettings.embed_model = self.embed_model

        # 延迟初始化
        self._vector_store: Optional[MilvusVectorStore] = None
        self._index: Optional[VectorStoreIndex] = None

        logger.info(
            f"[VectorStore] Initialized - host: {self.host}:{self.port}, "
            f"collection: {self.collection_name}"
        )

    def _get_vector_store(self, overwrite: bool = False) -> MilvusVectorStore:
        """
        获取或创建 Milvus 向量存储。

        Parameters
        ----------
        overwrite : bool
            是否覆盖现有集合

        Returns
        -------
        MilvusVectorStore
            向量存储实例
        """
        if self._vector_store is None or overwrite:
            try:
                uri = f"http://{self.host}:{self.port}"

                self._vector_store = MilvusVectorStore(
                    uri=uri,
                    collection_name=self.collection_name,
                    dim=self.dim,
                    overwrite=overwrite,
                    embedding_field="vector",  # Match existing Milvus schema
                    text_key="text",  # Match existing Milvus schema
                )

                logger.info(
                    f"[VectorStore] Connected to Milvus: {uri}, "
                    f"collection: {self.collection_name}"
                )
            except Exception as e:
                logger.error(f"[VectorStore] Failed to connect to Milvus: {e}")
                raise

        return self._vector_store

    @property
    def vector_store(self) -> MilvusVectorStore:
        """获取向量存储实例。"""
        return self._get_vector_store()

    def _get_index(self, nodes: Optional[List[BaseNode]] = None) -> VectorStoreIndex:
        """
        获取或创建向量索引。

        Parameters
        ----------
        nodes : List[BaseNode], optional
            要添加的节点列表

        Returns
        -------
        VectorStoreIndex
            向量索引实例
        """
        if self._index is None:
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            if nodes:
                self._index = VectorStoreIndex(
                    nodes=nodes,
                    storage_context=storage_context,
                    embed_model=self.embed_model,
                )
            else:
                # 从现有存储创建索引
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    embed_model=self.embed_model,
                )

        return self._index

    def get_index(self) -> VectorStoreIndex:
        """
        获取向量索引（用于 RAG 管道）。

        Returns
        -------
        VectorStoreIndex
            可用于构建查询引擎的索引

        Example
        -------
        ```python
        index = manager.get_index()
        query_engine = index.as_query_engine()
        response = query_engine.query("问题")
        ```
        """
        return self._get_index()

    def add_nodes(
        self,
        nodes: List[BaseNode],
        overwrite: bool = False,
        trace=None,
    ) -> List[str]:
        """
        添加 LlamaIndex 节点到向量存储。

        Parameters
        ----------
        nodes : List[BaseNode]
            TextNode 或其他 BaseNode 列表

        overwrite : bool
            是否覆盖现有数据

        trace : Langfuse Trace, optional
            Langfuse 追踪对象

        Returns
        -------
        List[str]
            节点 ID 列表

        Example
        -------
        ```python
        nodes = [
            TextNode(text="文档内容1", metadata={"source": "doc1"}),
            TextNode(text="文档内容2", metadata={"source": "doc2"}),
        ]
        ids = manager.add_nodes(nodes)
        ```
        """
        if not nodes:
            return []

        total_chars = sum(len(n.get_content()) for n in nodes)
        logger.info(f"[VectorStore] ========== 开始向量嵌入 ==========")
        logger.info(f"[VectorStore] 📊 节点数量: {len(nodes)}")
        logger.info(f"[VectorStore] 📝 总字符数: {total_chars:,} 字符")
        logger.info(f"[VectorStore] 🔧 嵌入模型: {settings.EMBEDDING_MODEL}")
        logger.info(f"[VectorStore] 📦 Milvus 集合: {self.collection_name}")
        logger.info(f"[VectorStore] 🔄 覆盖模式: {'是' if overwrite else '否'}")

        # Langfuse 追踪
        langfuse = get_langfuse_service()
        span = None
        if trace:
            span = langfuse.create_span(
                trace,
                name="vector_store_add",
                input={
                    "num_nodes": len(nodes),
                    "overwrite": overwrite,
                    "total_chars": total_chars,
                },
                metadata={
                    "collection": self.collection_name,
                    "embedding_model": settings.EMBEDDING_MODEL,
                },
            )

        start_time = time.time()

        try:
            if overwrite:
                self._vector_store = None
                self._index = None

            # 创建或更新索引
            logger.info(f"[VectorStore] 🚀 开始生成嵌入向量...")
            storage_context = StorageContext.from_defaults(
                vector_store=self._get_vector_store(overwrite=overwrite)
            )

            self._index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=True,  # 显示进度
            )

            ids = [node.node_id for node in nodes]
            elapsed = time.time() - start_time
            embed_speed = len(nodes) / elapsed if elapsed > 0 else 0

            logger.info(f"[VectorStore] ========== 嵌入完成 ==========")
            logger.info(
                f"[VectorStore] ✅ 成功嵌入 {len(nodes)} 个节点 | "
                f"耗时: {elapsed:.2f} 秒 | "
                f"速度: {embed_speed:.1f} 节点/秒"
            )
            logger.info(f"[VectorStore] Added {len(nodes)} nodes successfully")

            # 结束 Span - 记录嵌入操作
            if span:
                langfuse.end_span(
                    span,
                    output={
                        "num_nodes_added": len(nodes),
                        "node_ids": ids[:5],  # 只记录前5个
                        "elapsed_seconds": round(elapsed, 3),
                    },
                )

            # 记录嵌入操作到 Langfuse
            if trace and langfuse.enabled:
                langfuse.log_embedding(
                    trace=trace,
                    name="document_embedding",
                    model=settings.EMBEDDING_MODEL,
                    texts=[n.get_content()[:100] for n in nodes[:5]],
                    metadata={
                        "total_texts": len(nodes),
                        "collection": self.collection_name,
                    },
                )

            return ids

        except Exception as e:
            logger.error(f"[VectorStore] Failed to add nodes: {e}", exc_info=True)
            if span:
                langfuse.end_span(
                    span,
                    output={"error": str(e)},
                    level="ERROR",
                    status_message=str(e),
                )
            raise

    def add_documents(
        self,
        documents: List[LangchainDocument],
        overwrite: bool = False,
    ) -> List[str]:
        """
        添加 LangChain 文档到向量存储（自动转换）。

        保持与 LangChain 的兼容性。

        Parameters
        ----------
        documents : List[LangchainDocument]
            LangChain Document 列表

        overwrite : bool
            是否覆盖现有数据

        Returns
        -------
        List[str]
            文档 ID 列表

        Example
        -------
        ```python
        from langchain_core.documents import Document

        docs = [
            Document(page_content="内容1", metadata={"source": "doc1"}),
            Document(page_content="内容2", metadata={"source": "doc2"}),
        ]
        ids = manager.add_documents(docs)
        ```
        """
        if not documents:
            return []

        logger.info(
            f"[VectorStore] Converting {len(documents)} LangChain documents to nodes"
        )

        # 转换为 LlamaIndex TextNode
        nodes = [
            TextNode(
                text=doc.page_content,
                metadata=dict(doc.metadata) if doc.metadata else {},
            )
            for doc in documents
        ]

        return self.add_nodes(nodes, overwrite=overwrite)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        trace=None,
    ) -> List[NodeWithScore]:
        """
        检索相似节点。

        Parameters
        ----------
        query : str
            查询文本

        top_k : int
            返回结果数量

        trace : Langfuse Trace, optional
            Langfuse 追踪对象

        Returns
        -------
        List[NodeWithScore]
            带评分的节点列表

        Example
        -------
        ```python
        results = manager.retrieve("如何申请退款？", top_k=5)
        for node in results:
            print(f"Score: {node.score:.4f}")
            print(f"Content: {node.node.get_content()[:100]}")
            print(f"Source: {node.node.metadata.get('source')}")
        ```
        """
        logger.info(f"[VectorStore] Retrieving - query: {query[:50]}..., k: {top_k}")

        # Langfuse 追踪
        langfuse = get_langfuse_service()
        span = None
        if trace:
            span = langfuse.create_span(
                trace,
                name="vector_retrieval",
                input={
                    "query": query,
                    "top_k": top_k,
                },
                metadata={
                    "collection": self.collection_name,
                    "embedding_model": settings.EMBEDDING_MODEL,
                },
            )

        start_time = time.time()

        try:
            index = self._get_index()
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=top_k,
            )

            results = retriever.retrieve(query)
            elapsed = time.time() - start_time

            logger.info(f"[VectorStore] Retrieved {len(results)} nodes")
            for i, node in enumerate(results):
                source = node.node.metadata.get("source", "unknown")
                logger.debug(
                    f"[VectorStore] Node {i+1}: score={node.score:.4f}, "
                    f"source={source}"
                )

            # 结束 Span
            if span:
                langfuse.end_span(
                    span,
                    output={
                        "num_results": len(results),
                        "scores": [round(r.score, 4) for r in results],
                        "sources": [
                            r.node.metadata.get("source", "unknown")
                            for r in results
                        ],
                        "elapsed_seconds": round(elapsed, 3),
                    },
                )

            # 记录检索操作
            if trace and langfuse.enabled:
                langfuse.log_retrieval(
                    trace=trace,
                    name="vector_search",
                    query=query,
                    documents=[
                        {
                            "content": r.node.get_content()[:200],
                            "score": r.score,
                            "source": r.node.metadata.get("source", "unknown"),
                        }
                        for r in results
                    ],
                    metadata={
                        "top_k": top_k,
                        "collection": self.collection_name,
                    },
                )

            return results

        except Exception as e:
            logger.error(f"[VectorStore] Retrieve failed: {e}", exc_info=True)
            if span:
                langfuse.end_span(
                    span,
                    output={"error": str(e)},
                    level="ERROR",
                    status_message=str(e),
                )
            return []

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[LangchainDocument]:
        """
        LangChain 兼容的相似度搜索。

        返回 LangChain Document 格式，保持与旧代码的兼容性。

        Parameters
        ----------
        query : str
            查询文本

        k : int
            返回结果数量

        filter_dict : dict, optional
            元数据过滤条件（暂不支持）

        Returns
        -------
        List[LangchainDocument]
            LangChain Document 列表
        """
        logger.info(f"[VectorStore] similarity_search - query: {query[:50]}..., k: {k}")

        if filter_dict:
            logger.warning("[VectorStore] filter_dict not supported yet, ignoring")

        results = self.retrieve(query, top_k=k)

        # 转换为 LangChain Document
        documents = []
        for node_with_score in results:
            node = node_with_score.node
            doc = LangchainDocument(
                page_content=node.get_content(),
                metadata={
                    **dict(node.metadata),
                    "score": node_with_score.score,
                },
            )
            documents.append(doc)

        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> List[tuple]:
        """
        带评分的相似度搜索（LangChain 兼容）。

        Returns
        -------
        List[tuple]
            (Document, score) 元组列表
        """
        logger.info(
            f"[VectorStore] similarity_search_with_score - query: {query[:50]}..., k: {k}"
        )

        results = self.retrieve(query, top_k=k)

        return [
            (
                LangchainDocument(
                    page_content=node.node.get_content(),
                    metadata=dict(node.node.metadata),
                ),
                node.score,
            )
            for node in results
        ]

    def get_retriever(self, search_kwargs: Optional[Dict] = None) -> "VectorStoreRetriever":
        """
        获取 LangChain 兼容的检索器。

        Returns a proper BaseRetriever subclass for LCEL chain compatibility.

        Parameters
        ----------
        search_kwargs : dict, optional
            搜索参数，如 {"k": 5}

        Returns
        -------
        VectorStoreRetriever
            LangChain BaseRetriever 实现
        """
        kwargs = search_kwargs or {"k": 5}
        k = kwargs.get("k", 5)

        return VectorStoreRetriever(manager=self, top_k=k)

    def delete_collection(self) -> bool:
        """
        删除集合。

        Returns
        -------
        bool
            是否成功
        """
        try:
            # 重新创建以清空数据
            self._vector_store = None
            self._index = None
            self._get_vector_store(overwrite=True)
            logger.info(f"[VectorStore] Collection {self.collection_name} deleted")
            return True
        except Exception as e:
            logger.error(f"[VectorStore] Failed to delete collection: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息。

        Returns
        -------
        dict
            包含集合名称、实体数等信息
        """
        try:
            from pymilvus import Collection, connections

            connections.connect(host=self.host, port=self.port)
            collection = Collection(self.collection_name)
            collection.load()

            stats = {
                "name": self.collection_name,
                "num_entities": collection.num_entities,
                "schema": str(collection.schema),
            }
            return stats

        except Exception as e:
            logger.error(f"[VectorStore] Failed to get stats: {e}")
            return {"error": str(e)}


# ==================== 全局实例 ====================

_vector_store_manager: Optional[VectorStoreManager] = None


def get_vector_store_manager() -> VectorStoreManager:
    """
    获取全局向量存储管理器实例。

    Returns
    -------
    VectorStoreManager
        向量存储管理器单例
    """
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager
