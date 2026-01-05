"""
Document Processor / 文档处理器
================================

基于 LlamaIndex 的多格式文档加载和分块处理模块。

核心功能
--------
1. **多格式支持**: PDF, Word, Excel, Text, Markdown, HTML
2. **智能分块**: 递归字符分割 / 语义分块
3. **元数据提取**: 自动提取文件信息
4. **双格式兼容**: 同时支持 LlamaIndex 和 LangChain Document
5. **Langfuse 追踪**: 自动记录文档处理过程

分块策略
--------
1. **递归字符分割 (默认)**
   - 按固定大小分割
   - 支持自定义分隔符优先级
   - 适合通用文档

2. **语义分块 (可选)**
   - 基于语义边界分割
   - 保持段落完整性
   - 适合长文档和技术文档

Langfuse 追踪
-------------
```
Trace: document_processing
├── Span: load_file / load_directory
└── Span: split_documents
    ├── chunk_count
    ├── avg_chunk_size
    └── processing_time
```

配置参数
--------
```python
# config/settings.py
RAG_CHUNK_SIZE = 500              # 分块大小
RAG_CHUNK_OVERLAP = 50            # 重叠大小
RAG_USE_SEMANTIC_CHUNKING = True  # 使用语义分块
LANGFUSE_ENABLED = True           # 启用追踪
```

使用示例
--------
```python
from app.core.document_processor import DocumentProcessor

# 创建处理器
processor = DocumentProcessor(
    chunk_size=500,
    chunk_overlap=50,
    use_semantic_chunking=True
)

# 处理单个文件
docs = processor.process_file("/path/to/document.pdf")

# 处理目录
docs = processor.process_directory("/path/to/knowledge_base/")

# 处理文本
docs = processor.process_text("这是一段文本内容...")

# 获取 LangChain 格式 (用于向量存储)
langchain_docs = processor.to_langchain_documents(docs)
```

Author: Intelligent Customer Service Team
Version: 2.1.0 (LlamaIndex + Langfuse)
"""
import logging
import time
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from llama_index.core import Document as LlamaDocument
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.core.schema import TextNode, NodeWithScore

# LangChain Document for compatibility
from langchain_core.documents import Document as LangchainDocument

from config.settings import settings

# Langfuse observability
from app.services.langfuse_service import get_langfuse_service

logger = logging.getLogger(__name__)

# Try to import optional readers
PDF_READER_AVAILABLE = False
DOCX_READER_AVAILABLE = False
MARKDOWN_READER_AVAILABLE = False
CSV_READER_AVAILABLE = False
HTML_READER_AVAILABLE = False
EXCEL_READER_AVAILABLE = False

try:
    from llama_index.readers.file import PDFReader
    PDF_READER_AVAILABLE = True
except ImportError:
    logger.debug("[DocumentProcessor] PDFReader not available, using default")

try:
    from llama_index.readers.file import DocxReader
    DOCX_READER_AVAILABLE = True
except ImportError:
    logger.debug("[DocumentProcessor] DocxReader not available, using default")

try:
    from llama_index.readers.file import MarkdownReader
    MARKDOWN_READER_AVAILABLE = True
except ImportError:
    logger.debug("[DocumentProcessor] MarkdownReader not available, using default")

try:
    from llama_index.readers.file import CSVReader
    CSV_READER_AVAILABLE = True
except ImportError:
    logger.debug("[DocumentProcessor] CSVReader not available, using default")

try:
    from llama_index.readers.file import HTMLTagReader
    HTML_READER_AVAILABLE = True
except ImportError:
    logger.debug("[DocumentProcessor] HTMLTagReader not available, using default")

try:
    from llama_index.readers.file import PandasExcelReader
    EXCEL_READER_AVAILABLE = True
except ImportError:
    logger.debug("[DocumentProcessor] PandasExcelReader not available, using default")


def _build_file_extractors() -> dict:
    """
    构建文件扩展名到 Reader 的映射。

    Returns
    -------
    dict
        文件扩展名到 Reader 实例的映射
    """
    extractors = {}

    if DOCX_READER_AVAILABLE:
        extractors[".docx"] = DocxReader()
        extractors[".doc"] = DocxReader()

    if PDF_READER_AVAILABLE:
        extractors[".pdf"] = PDFReader()

    if MARKDOWN_READER_AVAILABLE:
        extractors[".md"] = MarkdownReader()

    if CSV_READER_AVAILABLE:
        extractors[".csv"] = CSVReader()

    if HTML_READER_AVAILABLE:
        extractors[".html"] = HTMLTagReader()
        extractors[".htm"] = HTMLTagReader()

    if EXCEL_READER_AVAILABLE:
        extractors[".xlsx"] = PandasExcelReader()
        extractors[".xls"] = PandasExcelReader()

    logger.info(f"[DocumentProcessor] File extractors configured: {list(extractors.keys())}")
    return extractors


# Global file extractors instance
FILE_EXTRACTORS = _build_file_extractors()


class DocumentProcessor:
    """
    文档处理器 (LlamaIndex 实现)
    ===========================

    提供多格式文档的加载、分块和转换功能。

    Attributes
    ----------
    chunk_size : int
        分块大小（字符数），默认 500

    chunk_overlap : int
        分块重叠大小，默认 50

    use_semantic_chunking : bool
        是否使用语义分块，默认 False

    SUPPORTED_EXTENSIONS : dict
        支持的文件扩展名映射

    Example
    -------
    ```python
    processor = DocumentProcessor()

    # 加载 PDF
    docs = processor.load_file("manual.pdf")

    # 分块
    chunks = processor.split_documents(docs)

    # 转换为 LangChain 格式
    lc_docs = processor.to_langchain_documents(chunks)
    ```
    """

    SUPPORTED_EXTENSIONS = {
        ".txt": "text",
        ".md": "markdown",
        ".pdf": "pdf",
        ".d"
        "ocx": "word",
        ".doc": "word",
        ".xlsx": "excel",
        ".xls": "excel",
        ".html": "html",
        ".htm": "html",
        ".csv": "csv",
        ".json": "json",
    }

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        use_semantic_chunking: bool = None,
    ):
        """
        初始化文档处理器。

        Parameters
        ----------
        chunk_size : int, optional
            分块大小，默认使用 settings.RAG_CHUNK_SIZE

        chunk_overlap : int, optional
            分块重叠，默认使用 settings.RAG_CHUNK_OVERLAP

        use_semantic_chunking : bool, optional
            是否使用语义分块，默认使用 settings.RAG_USE_SEMANTIC_CHUNKING
        """
        self.chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP
        self.use_semantic_chunking = (
            use_semantic_chunking
            if use_semantic_chunking is not None
            else settings.RAG_USE_SEMANTIC_CHUNKING
        )

        # 初始化分块器
        self._init_splitters()

        logger.info(
            f"[DocumentProcessor] Initialized - chunk_size: {self.chunk_size}, "
            f"overlap: {self.chunk_overlap}, semantic: {self.use_semantic_chunking}"
        )

    def _init_splitters(self):
        """初始化文本分块器。"""
        # 句子分块器 (默认)
        self.sentence_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # 中英文分隔符
            paragraph_separator="\n\n",
            secondary_chunking_regex="[。！？.!?]",
        )

        # 语义分块器 (可选，需要嵌入模型)
        self._semantic_splitter = None

    def _get_semantic_splitter(self):
        """懒加载语义分块器。"""
        if self._semantic_splitter is None:
            try:
                from llama_index.core import Settings as LlamaSettings

                if LlamaSettings.embed_model is not None:
                    self._semantic_splitter = SemanticSplitterNodeParser(
                        embed_model=LlamaSettings.embed_model,
                        buffer_size=settings.RAG_SEMANTIC_CHUNK_BUFFER_SIZE,
                        breakpoint_percentile_threshold=95,
                    )
                    logger.info("[DocumentProcessor] Semantic splitter initialized")
                else:
                    logger.warning(
                        "[DocumentProcessor] No embed model, falling back to sentence splitter"
                    )
            except Exception as e:
                logger.warning(f"[DocumentProcessor] Semantic splitter init failed: {e}")

        return self._semantic_splitter

    def load_file(
        self,
        file_path: str,
        trace=None,
    ) -> List[LlamaDocument]:
        """
        加载单个文件。

        Parameters
        ----------
        file_path : str
            文件路径

        trace : Langfuse Trace, optional
            Langfuse 追踪对象

        Returns
        -------
        List[LlamaDocument]
            LlamaIndex Document 列表

        Raises
        ------
        FileNotFoundError
            文件不存在

        ValueError
            不支持的文件类型

        Example
        -------
        ```python
        docs = processor.load_file("/path/to/file.pdf")
        print(f"Loaded {len(docs)} documents")
        print(docs[0].text[:100])
        ```
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        file_type = self.SUPPORTED_EXTENSIONS[ext]
        logger.info(f"[DocumentProcessor] Loading {file_type} file: {path.name}")

        # Langfuse 追踪
        langfuse = get_langfuse_service()
        span = None
        if trace:
            span = langfuse.create_span(
                trace,
                name="load_file",
                input={
                    "file_path": path.name,
                    "file_type": file_type,
                    "file_size": path.stat().st_size,
                },
            )

        start_time = time.time()

        try:
            # 使用 SimpleDirectoryReader 加载单个文件
            reader = SimpleDirectoryReader(
                input_files=[str(path)],
                filename_as_id=True,
                file_extractor=FILE_EXTRACTORS,  # 使用显式配置的文件解析器
            )
            documents = reader.load_data()

            # 添加元数据
            for doc in documents:
                doc.metadata["source"] = path.name
                doc.metadata["file_type"] = file_type
                doc.metadata["file_path"] = str(path.absolute())

            elapsed = time.time() - start_time
            logger.info(f"[DocumentProcessor] Loaded {len(documents)} document(s)")

            # 结束 Span
            if span:
                langfuse.end_span(
                    span,
                    output={
                        "num_documents": len(documents),
                        "total_chars": sum(len(doc.text) for doc in documents),
                        "elapsed_seconds": round(elapsed, 3),
                    },
                )

            return documents

        except Exception as e:
            logger.error(f"[DocumentProcessor] Failed to load {file_path}: {e}")
            if span:
                langfuse.end_span(
                    span,
                    output={"error": str(e)},
                    level="ERROR",
                    status_message=str(e),
                )
            raise

    def load_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        exclude_hidden: bool = True,
    ) -> List[LlamaDocument]:
        """
        加载目录下所有支持的文件。

        Parameters
        ----------
        directory_path : str
            目录路径

        recursive : bool
            是否递归子目录，默认 True

        exclude_hidden : bool
            是否排除隐藏文件，默认 True

        Returns
        -------
        List[LlamaDocument]
            所有文档列表

        Example
        -------
        ```python
        docs = processor.load_directory(
            "/path/to/knowledge_base",
            recursive=True
        )
        ```
        """
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")

        logger.info(f"[DocumentProcessor] Loading directory: {directory_path}")

        try:
            # 构建支持的扩展名列表
            required_exts = list(self.SUPPORTED_EXTENSIONS.keys())

            reader = SimpleDirectoryReader(
                input_dir=str(path),
                recursive=recursive,
                exclude_hidden=exclude_hidden,
                required_exts=required_exts,
                filename_as_id=True,
                file_extractor=FILE_EXTRACTORS,  # 使用显式配置的文件解析器
            )
            documents = reader.load_data()

            # 添加元数据并输出处理的文档信息
            logger.info(f"[DocumentProcessor] ========== 文档加载详情 ==========")
            for i, doc in enumerate(documents, 1):
                file_path = doc.metadata.get("file_path", "")
                if file_path:
                    p = Path(file_path)
                    doc.metadata["source"] = p.name
                    doc.metadata["file_type"] = self.SUPPORTED_EXTENSIONS.get(
                        p.suffix.lower(), "unknown"
                    )
                    file_size = p.stat().st_size if p.exists() else 0
                    logger.info(
                        f"[DocumentProcessor] 📄 文档 {i}/{len(documents)}: {p.name} "
                        f"| 类型: {doc.metadata['file_type']} "
                        f"| 大小: {file_size/1024:.1f} KB "
                        f"| 内容长度: {len(doc.text)} 字符"
                    )

            logger.info(f"[DocumentProcessor] ========== 加载完成 ==========")
            logger.info(
                f"[DocumentProcessor] ✅ 共加载 {len(documents)} 个文档"
            )
            return documents

        except Exception as e:
            logger.error(f"[DocumentProcessor] Failed to load directory: {e}")
            raise

    def split_documents(
        self,
        documents: List[LlamaDocument],
        use_semantic: bool = None,
        trace=None,
    ) -> List[TextNode]:
        """
        将文档分割为文本节点。

        Parameters
        ----------
        documents : List[LlamaDocument]
            待分割的文档列表

        use_semantic : bool, optional
            是否使用语义分块，默认使用实例配置

        trace : Langfuse Trace, optional
            Langfuse 追踪对象

        Returns
        -------
        List[TextNode]
            分割后的文本节点列表

        Note
        ----
        语义分块需要嵌入模型支持，如果不可用会自动降级到句子分块。

        Example
        -------
        ```python
        docs = processor.load_file("document.pdf")
        nodes = processor.split_documents(docs)
        print(f"Split into {len(nodes)} chunks")
        ```
        """
        if not documents:
            return []

        use_semantic = use_semantic if use_semantic is not None else self.use_semantic_chunking

        logger.info(
            f"[DocumentProcessor] Splitting {len(documents)} documents "
            f"(semantic={use_semantic})"
        )

        # Langfuse 追踪
        langfuse = get_langfuse_service()
        span = None
        if trace:
            span = langfuse.create_span(
                trace,
                name="split_documents",
                input={
                    "num_documents": len(documents),
                    "total_chars": sum(len(doc.text) for doc in documents),
                    "use_semantic": use_semantic,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
            )

        start_time = time.time()

        try:
            if use_semantic:
                splitter = self._get_semantic_splitter()
                if splitter is None:
                    splitter = self.sentence_splitter
                    logger.info("[DocumentProcessor] ⚠️ 语义分块器不可用，回退到句子分块器")
            else:
                splitter = self.sentence_splitter

            splitter_type = "语义分块" if use_semantic and splitter != self.sentence_splitter else "句子分块"
            logger.info(f"[DocumentProcessor] ========== 开始文档分块 ==========")
            logger.info(f"[DocumentProcessor] 🔧 分块策略: {splitter_type}")
            logger.info(f"[DocumentProcessor] 📏 分块大小: {self.chunk_size} 字符, 重叠: {self.chunk_overlap} 字符")

            # 分割文档
            nodes = splitter.get_nodes_from_documents(documents)

            elapsed = time.time() - start_time
            avg_size = sum(len(n.text) for n in nodes) // max(len(nodes), 1)
            min_size = min(len(n.text) for n in nodes) if nodes else 0
            max_size = max(len(n.text) for n in nodes) if nodes else 0

            # 输出每个文档的分块详情
            logger.info(f"[DocumentProcessor] ========== 分块详情 ==========")
            doc_chunks = {}
            for node in nodes:
                source = node.metadata.get("source", "unknown")
                if source not in doc_chunks:
                    doc_chunks[source] = []
                doc_chunks[source].append(len(node.text))

            for source, chunk_sizes in doc_chunks.items():
                logger.info(
                    f"[DocumentProcessor] 📄 {source}: "
                    f"{len(chunk_sizes)} 个分块 | "
                    f"平均: {sum(chunk_sizes)//len(chunk_sizes)} 字符 | "
                    f"范围: {min(chunk_sizes)}-{max(chunk_sizes)} 字符"
                )

            logger.info(f"[DocumentProcessor] ========== 分块完成 ==========")
            logger.info(
                f"[DocumentProcessor] ✅ 共生成 {len(nodes)} 个分块 | "
                f"平均: {avg_size} 字符 | 范围: {min_size}-{max_size} 字符 | "
                f"耗时: {elapsed:.2f} 秒"
            )

            # 结束 Span
            if span:
                langfuse.end_span(
                    span,
                    output={
                        "num_chunks": len(nodes),
                        "avg_chunk_size": avg_size,
                        "min_chunk_size": min(len(n.text) for n in nodes) if nodes else 0,
                        "max_chunk_size": max(len(n.text) for n in nodes) if nodes else 0,
                        "elapsed_seconds": round(elapsed, 3),
                        "splitter_type": "semantic" if use_semantic else "sentence",
                    },
                )

            return nodes

        except Exception as e:
            logger.error(f"[DocumentProcessor] Split failed: {e}")
            if span:
                langfuse.end_span(
                    span,
                    output={"error": str(e)},
                    level="ERROR",
                    status_message=str(e),
                )
            # 降级处理：直接返回文档内容作为单个节点
            return [
                TextNode(text=doc.text, metadata=doc.metadata)
                for doc in documents
            ]

    def process_file(
        self,
        file_path: str,
        split: bool = True,
        enable_trace: bool = True,
    ) -> Union[List[LlamaDocument], List[TextNode]]:
        """
        加载并处理单个文件。

        Parameters
        ----------
        file_path : str
            文件路径

        split : bool
            是否分块，默认 True

        enable_trace : bool
            是否启用 Langfuse 追踪，默认 True

        Returns
        -------
        Union[List[LlamaDocument], List[TextNode]]
            如果 split=True 返回 TextNode 列表，否则返回 Document 列表

        Example
        -------
        ```python
        # 加载并分块
        nodes = processor.process_file("document.pdf", split=True)

        # 仅加载不分块
        docs = processor.process_file("document.pdf", split=False)
        ```
        """
        # 创建 Langfuse 追踪
        langfuse = get_langfuse_service()
        trace = None

        if enable_trace and langfuse.enabled:
            path = Path(file_path)
            trace = langfuse.create_trace(
                name="document_processing",
                input={
                    "file_path": path.name,
                    "split": split,
                },
                metadata={
                    "file_type": path.suffix.lower(),
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "use_semantic_chunking": self.use_semantic_chunking,
                },
                tags=["document_processing", path.suffix.lower().lstrip(".")],
            )

        try:
            documents = self.load_file(file_path, trace=trace)

            if split:
                nodes = self.split_documents(documents, trace=trace)

                # 结束追踪
                if trace:
                    langfuse.end_trace(
                        trace,
                        output={
                            "status": "success",
                            "num_documents": len(documents),
                            "num_chunks": len(nodes),
                        },
                    )
                return nodes

            if trace:
                langfuse.end_trace(
                    trace,
                    output={
                        "status": "success",
                        "num_documents": len(documents),
                    },
                )
            return documents

        except Exception as e:
            if trace:
                langfuse.end_trace(
                    trace,
                    output={"status": "error", "error": str(e)},
                    metadata={"error_type": type(e).__name__},
                )
            raise

    def process_directory(
        self,
        directory_path: str,
        split: bool = True,
        recursive: bool = True,
    ) -> Union[List[LlamaDocument], List[TextNode]]:
        """
        加载并处理目录下所有文件。

        Parameters
        ----------
        directory_path : str
            目录路径

        split : bool
            是否分块，默认 True

        recursive : bool
            是否递归，默认 True

        Returns
        -------
        Union[List[LlamaDocument], List[TextNode]]
            处理后的文档或节点列表
        """
        documents = self.load_directory(directory_path, recursive=recursive)
        if split:
            return self.split_documents(documents)
        return documents

    def process_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        split: bool = True,
    ) -> Union[List[LlamaDocument], List[TextNode]]:
        """
        处理原始文本。

        Parameters
        ----------
        text : str
            原始文本内容

        metadata : dict, optional
            元数据字典

        split : bool
            是否分块，默认 True

        Returns
        -------
        Union[List[LlamaDocument], List[TextNode]]
            处理后的文档或节点列表

        Example
        -------
        ```python
        nodes = processor.process_text(
            "这是一段很长的文本...",
            metadata={"source": "user_input", "category": "FAQ"}
        )
        ```
        """
        doc = LlamaDocument(
            text=text,
            metadata=metadata or {"source": "direct_input"},
        )
        documents = [doc]

        if split:
            return self.split_documents(documents)
        return documents

    # ==================== 格式转换 ====================

    def to_langchain_documents(
        self,
        nodes: Union[List[TextNode], List[LlamaDocument], List[NodeWithScore]],
    ) -> List[LangchainDocument]:
        """
        将 LlamaIndex 节点/文档转换为 LangChain Document。

        用于与现有的向量存储和工具链集成。

        Parameters
        ----------
        nodes : List
            LlamaIndex TextNode、Document 或 NodeWithScore 列表

        Returns
        -------
        List[LangchainDocument]
            LangChain Document 列表

        Example
        -------
        ```python
        # LlamaIndex 节点
        nodes = processor.process_file("doc.pdf")

        # 转换为 LangChain 格式
        lc_docs = processor.to_langchain_documents(nodes)

        # 用于向量存储
        vector_store.add_documents(lc_docs)
        ```
        """
        langchain_docs = []

        for item in nodes:
            # 处理 NodeWithScore
            if isinstance(item, NodeWithScore):
                node = item.node
                text = node.get_content()
                metadata = dict(node.metadata) if node.metadata else {}
                metadata["score"] = item.score
            # 处理 TextNode
            elif isinstance(item, TextNode):
                text = item.text
                metadata = dict(item.metadata) if item.metadata else {}
            # 处理 LlamaDocument
            elif isinstance(item, LlamaDocument):
                text = item.text
                metadata = dict(item.metadata) if item.metadata else {}
            else:
                logger.warning(f"[DocumentProcessor] Unknown type: {type(item)}")
                continue

            langchain_docs.append(
                LangchainDocument(page_content=text, metadata=metadata)
            )

        return langchain_docs

    def from_langchain_documents(
        self,
        documents: List[LangchainDocument],
    ) -> List[LlamaDocument]:
        """
        将 LangChain Document 转换为 LlamaIndex Document。

        Parameters
        ----------
        documents : List[LangchainDocument]
            LangChain Document 列表

        Returns
        -------
        List[LlamaDocument]
            LlamaIndex Document 列表
        """
        return [
            LlamaDocument(
                text=doc.page_content,
                metadata=dict(doc.metadata) if doc.metadata else {},
            )
            for doc in documents
        ]

    # ==================== 工具方法 ====================

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """获取支持的文件扩展名列表。"""
        return list(DocumentProcessor.SUPPORTED_EXTENSIONS.keys())

    @staticmethod
    def is_supported(file_path: str) -> bool:
        """
        检查文件是否支持处理。

        Parameters
        ----------
        file_path : str
            文件路径

        Returns
        -------
        bool
            是否支持
        """
        ext = Path(file_path).suffix.lower()
        return ext in DocumentProcessor.SUPPORTED_EXTENSIONS


# ==================== 全局实例 ====================

_document_processor: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    """
    获取全局文档处理器实例。

    Returns
    -------
    DocumentProcessor
        文档处理器单例

    Example
    -------
    ```python
    from app.core.document_processor import get_document_processor

    processor = get_document_processor()
    docs = processor.process_file("document.pdf")
    ```
    """
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor
