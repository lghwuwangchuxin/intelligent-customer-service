"""
Upload Service / 文件上传服务
==============================

批量文件上传处理模块，支持进度追踪和实时状态更新。

核心功能
--------
1. **批量上传**: 支持多文件并发处理
2. **进度追踪**: 单文件和整体进度实时更新
3. **格式验证**: 自动校验文件类型和大小
4. **异步处理**: SSE 流式进度推送

支持格式
--------
- PDF (.pdf)
- Word (.docx, .doc)
- Excel (.xlsx, .xls)
- 文本 (.txt, .md)
- 网页 (.html, .htm)
- 数据 (.csv, .json)

使用示例
--------
```python
from app.services.upload_service import UploadService

# 初始化
upload_service = UploadService(rag_service=rag_service)

# 创建批量任务
task = upload_service.create_batch_task([
    {"filename": "doc1.pdf", "file_size": 1024},
    {"filename": "doc2.docx", "file_size": 2048},
])

# 处理文件（异步生成进度）
async for progress in upload_service.process_batch(task.id, files):
    print(progress)
```

Author: Intelligent Customer Service Team
Version: 2.0.0
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncIterator
from pathlib import Path
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UploadStatus(str, Enum):
    """Upload task status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FileUploadInfo(BaseModel):
    """Information about a single file upload."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_size: int
    file_type: str
    status: UploadStatus = UploadStatus.PENDING
    progress: int = 0  # 0-100
    chunks_processed: int = 0
    total_chunks: int = 0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class BatchUploadTask(BaseModel):
    """Batch upload task tracking."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    files: List[FileUploadInfo] = Field(default_factory=list)
    status: UploadStatus = UploadStatus.PENDING
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def progress(self) -> int:
        """Calculate overall progress."""
        if self.total_files == 0:
            return 0
        return int((self.completed_files + self.failed_files) / self.total_files * 100)


class UploadService:
    """
    文件上传服务 (Upload Service)
    ============================

    处理文件上传的核心服务，支持批量处理和进度追踪。

    功能特性
    --------
    - **批量处理**: 支持多文件同时上传
    - **进度追踪**: 单文件和整体进度实时更新
    - **异步处理**: 非阻塞文件处理
    - **SSE 推送**: 流式进度更新

    与 LlamaIndex DocumentProcessor 集成
    ------------------------------------
    上传的文件会通过 RAGService 处理:
    1. DocumentProcessor 加载和分块文件 (LlamaIndex)
    2. 转换为 LangChain Document 格式
    3. 存入 Milvus 向量数据库

    Attributes
    ----------
    rag_service : RAGService
        RAG 服务实例，用于文档处理和索引

    upload_dir : Path
        临时文件存储目录

    max_file_size : int
        最大文件大小（字节）

    Example
    -------
    ```python
    service = UploadService(rag_service)

    # 验证文件
    valid, error = service.validate_file("doc.pdf", 1024000)

    # 创建任务
    task = service.create_batch_task([...])

    # 处理并获取进度
    async for update in service.process_batch(task.id, files):
        print(f"Progress: {update['progress']}%")
    ```
    """

    # 支持的文件类型 (与 DocumentProcessor 保持同步)
    SUPPORTED_EXTENSIONS = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
        ".csv": "text/csv",
        ".json": "application/json",
    }

    def __init__(
        self,
        rag_service,
        upload_dir: str = "./uploads",
        max_file_size: int = 50 * 1024 * 1024,  # 50MB
    ):
        """
        Initialize upload service.

        Args:
            rag_service: RAG service for document processing.
            upload_dir: Directory for temporary file storage.
            max_file_size: Maximum file size in bytes.
        """
        self.rag_service = rag_service
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size

        # In-memory task storage (could be replaced with Redis/DB)
        self._tasks: Dict[str, BatchUploadTask] = {}

    def validate_file(self, filename: str, file_size: int) -> tuple[bool, Optional[str]]:
        """
        Validate a file for upload.

        Args:
            filename: Name of the file.
            file_size: Size of the file in bytes.

        Returns:
            Tuple of (is_valid, error_message).
        """
        ext = Path(filename).suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            return False, f"Unsupported file type: {ext}. Supported: {list(self.SUPPORTED_EXTENSIONS.keys())}"

        if file_size > self.max_file_size:
            max_mb = self.max_file_size / (1024 * 1024)
            return False, f"File too large. Maximum size: {max_mb}MB"

        return True, None

    def create_batch_task(self, file_infos: List[Dict[str, Any]]) -> BatchUploadTask:
        """
        Create a batch upload task.

        Args:
            file_infos: List of file information dicts with filename and file_size.

        Returns:
            BatchUploadTask instance.
        """
        files = []
        for info in file_infos:
            filename = info.get("filename", "unknown")
            file_size = info.get("file_size", 0)
            ext = Path(filename).suffix.lower()

            file_info = FileUploadInfo(
                filename=filename,
                file_size=file_size,
                file_type=self.SUPPORTED_EXTENSIONS.get(ext, "application/octet-stream"),
            )

            # Validate
            is_valid, error = self.validate_file(filename, file_size)
            if not is_valid:
                file_info.status = UploadStatus.FAILED
                file_info.error_message = error

            files.append(file_info)

        task = BatchUploadTask(
            files=files,
            total_files=len(files),
        )

        self._tasks[task.id] = task
        logger.info(f"Created batch upload task: {task.id} with {len(files)} files")

        return task

    def get_task(self, task_id: str) -> Optional[BatchUploadTask]:
        """Get a batch upload task by ID."""
        return self._tasks.get(task_id)

    async def process_file(
        self,
        task_id: str,
        file_id: str,
        file_content: bytes,
    ) -> FileUploadInfo:
        """
        处理单个文件上传。

        处理流程
        --------
        1. 保存文件到临时目录
        2. 调用 RAGService.add_knowledge() 处理文件
           - LlamaIndex DocumentProcessor 加载和分块
           - 转换为 LangChain Document
           - 存入向量数据库
        3. 清理临时文件
        4. 更新任务状态

        Parameters
        ----------
        task_id : str
            批量任务 ID

        file_id : str
            文件 ID

        file_content : bytes
            文件内容

        Returns
        -------
        FileUploadInfo
            更新后的文件信息

        Raises
        ------
        ValueError
            任务或文件不存在
        """
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        file_info = None
        for f in task.files:
            if f.id == file_id:
                file_info = f
                break

        if not file_info:
            raise ValueError(f"File {file_id} not found in task {task_id}")

        if file_info.status == UploadStatus.FAILED:
            return file_info

        try:
            file_info.status = UploadStatus.PROCESSING
            file_info.started_at = datetime.utcnow()
            task.status = UploadStatus.PROCESSING
            task.updated_at = datetime.utcnow()

            # 步骤 1: 保存临时文件
            temp_path = self.upload_dir / f"{file_id}_{file_info.filename}"
            with open(temp_path, "wb") as f:
                f.write(file_content)

            logger.info(f"[Upload] 处理文件: {file_info.filename} -> {temp_path}")

            # 步骤 2: 使用 RAG 服务处理（LlamaIndex + Milvus）
            file_info.progress = 30
            result = self.rag_service.add_knowledge(file_path=str(temp_path))

            if result.get("success"):
                file_info.status = UploadStatus.COMPLETED
                file_info.progress = 100
                # 优先使用 num_nodes（LlamaIndex 分块数），回退到 num_documents
                file_info.total_chunks = result.get("num_nodes", result.get("num_documents", 0))
                file_info.chunks_processed = file_info.total_chunks
                task.completed_files += 1

                logger.info(
                    f"[Upload] 文件处理成功: {file_info.filename}, "
                    f"分块数: {file_info.total_chunks}"
                )
            else:
                file_info.status = UploadStatus.FAILED
                file_info.error_message = result.get("error", "Unknown error")
                task.failed_files += 1

                logger.error(
                    f"[Upload] 文件处理失败: {file_info.filename}, "
                    f"错误: {file_info.error_message}"
                )

            file_info.completed_at = datetime.utcnow()

            # 步骤 3: 清理临时文件
            try:
                temp_path.unlink()
            except Exception:
                pass

        except Exception as e:
            logger.error(f"[Upload] 处理异常: {file_info.filename}: {e}", exc_info=True)
            file_info.status = UploadStatus.FAILED
            file_info.error_message = str(e)
            file_info.completed_at = datetime.utcnow()
            task.failed_files += 1

        # 步骤 4: 更新任务状态
        if task.completed_files + task.failed_files >= task.total_files:
            if task.failed_files == task.total_files:
                task.status = UploadStatus.FAILED
            elif task.failed_files > 0:
                task.status = UploadStatus.COMPLETED  # 部分成功
            else:
                task.status = UploadStatus.COMPLETED

        task.updated_at = datetime.utcnow()

        return file_info

    async def process_batch(
        self,
        task_id: str,
        files: List[tuple[str, bytes]],
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process multiple files and yield progress updates.

        Args:
            task_id: Batch task ID.
            files: List of (file_id, content) tuples.

        Yields:
            Progress update dicts.
        """
        task = self.get_task(task_id)
        if not task:
            yield {"type": "error", "message": f"Task {task_id} not found"}
            return

        yield {
            "type": "start",
            "task_id": task_id,
            "total_files": task.total_files,
        }

        for file_id, content in files:
            file_info = None
            for f in task.files:
                if f.id == file_id:
                    file_info = f
                    break

            if not file_info:
                continue

            yield {
                "type": "file_start",
                "file_id": file_id,
                "filename": file_info.filename,
            }

            try:
                result = await self.process_file(task_id, file_id, content)

                yield {
                    "type": "file_complete",
                    "file_id": file_id,
                    "filename": result.filename,
                    "status": result.status.value,
                    "chunks": result.chunks_processed,
                    "error": result.error_message,
                }
            except Exception as e:
                yield {
                    "type": "file_error",
                    "file_id": file_id,
                    "filename": file_info.filename,
                    "error": str(e),
                }

            yield {
                "type": "progress",
                "task_id": task_id,
                "progress": task.progress,
                "completed": task.completed_files,
                "failed": task.failed_files,
                "total": task.total_files,
            }

        yield {
            "type": "complete",
            "task_id": task_id,
            "status": task.status.value,
            "completed_files": task.completed_files,
            "failed_files": task.failed_files,
        }

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of a batch upload task.

        Args:
            task_id: Task ID.

        Returns:
            Task status dict or None if not found.
        """
        task = self.get_task(task_id)
        if not task:
            return None

        return {
            "task_id": task.id,
            "status": task.status.value,
            "progress": task.progress,
            "total_files": task.total_files,
            "completed_files": task.completed_files,
            "failed_files": task.failed_files,
            "files": [
                {
                    "id": f.id,
                    "filename": f.filename,
                    "status": f.status.value,
                    "progress": f.progress,
                    "chunks": f.chunks_processed,
                    "error": f.error_message,
                }
                for f in task.files
            ],
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
        }

    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed tasks.

        Args:
            max_age_hours: Maximum age in hours for completed tasks.

        Returns:
            Number of tasks cleaned up.
        """
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        to_remove = []

        for task_id, task in self._tasks.items():
            if task.status in (UploadStatus.COMPLETED, UploadStatus.FAILED):
                if task.updated_at < cutoff:
                    to_remove.append(task_id)

        for task_id in to_remove:
            del self._tasks[task_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old upload tasks")

        return len(to_remove)
