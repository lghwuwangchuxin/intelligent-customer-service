"""
Knowledge Endpoints - Handles knowledge base management.

Bounded Context: Knowledge Base Domain

Supports:
- Traditional vector-only search (Milvus)
- Hybrid search (ES BM25 + Milvus vectors) when ES is enabled
- Advanced metadata filtering via Elasticsearch
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.schemas import (
    BatchUploadRequest,
    BatchUploadResponse,
    KnowledgeAddRequest,
    KnowledgeResponse,
    SearchRequest,
    SearchResponse,
)
from app.infrastructure.factory import get_registry
from config.settings import settings

logger = logging.getLogger(__name__)


# ============== Additional Schemas for Hybrid Search ==============

class AdvancedSearchRequest(BaseModel):
    """Request for advanced hybrid search with metadata filtering."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters (e.g., {'file_type': 'pdf', 'source': 'manual'})"
    )
    vector_weight: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Weight for vector search results"
    )
    bm25_weight: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Weight for BM25 keyword search results"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include full metadata in results"
    )


class AdvancedSearchResult(BaseModel):
    """Single result from advanced search."""
    chunk_id: str
    content: str
    score: float
    source: str  # "milvus", "es_bm25", "hybrid"
    metadata: Dict[str, Any] = {}


class AdvancedSearchResponse(BaseModel):
    """Response from advanced hybrid search."""
    query: str
    results: List[AdvancedSearchResult]
    total_count: int
    search_type: str  # "hybrid" or "vector_only"
    weights: Dict[str, float] = {}

router = APIRouter(prefix="/api/knowledge", tags=["Knowledge Base"])

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".txt", ".md"}


@router.post("/upload", response_model=KnowledgeResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(None),
):
    """
    Upload a document to the knowledge base.

    Uses async embedding with optimizations:
    - Text preprocessing for better LLM understanding
    - Deduplication (skip already embedded content)
    - Auto-delete existing nodes when re-uploading same file

    Supported formats: PDF, Word (.docx), Excel (.xlsx), Text (.txt), Markdown (.md)
    """
    registry = get_registry()

    # Validate file type
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}",
        )

    try:
        # Save uploaded file temporarily
        temp_path = Path(settings.KNOWLEDGE_BASE_PATH) / filename
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Process and index the document using async embedding
        result = await registry.get("rag").async_add_knowledge(file_path=str(temp_path))

        if result["success"]:
            return KnowledgeResponse(
                success=True,
                message=f"Successfully indexed {filename}",
                num_documents=result.get("num_nodes"),
                details=result,
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error"))

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-text", response_model=KnowledgeResponse)
async def add_text_knowledge(request: KnowledgeAddRequest):
    """
    Add text content directly to the knowledge base.

    Uses async embedding with optimizations:
    - Text preprocessing for better LLM understanding
    - Deduplication (skip already embedded content)
    """
    registry = get_registry()

    try:
        metadata = request.metadata or {}
        if request.title:
            metadata["title"] = request.title

        result = await registry.get("rag").async_add_knowledge(
            text=request.text,
            metadata=metadata,
        )

        if result["success"]:
            return KnowledgeResponse(
                success=True,
                message="Successfully added text to knowledge base",
                num_documents=result.get("num_nodes"),
                details=result,
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error"))

    except Exception as e:
        logger.error(f"Add text error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search_knowledge(request: SearchRequest):
    """
    Search the knowledge base.
    """
    registry = get_registry()

    try:
        results = registry.get("rag").get_relevant_documents(request.query)
        return SearchResponse(
            results=results[:request.top_k],
            query=request.query,
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index-directory", response_model=KnowledgeResponse)
async def index_directory(directory_path: str = Form(...)):
    """
    Index all documents in a directory.
    """
    registry = get_registry()

    try:
        result = await registry.get("rag").async_index_directory(directory_path)

        if result["success"]:
            return KnowledgeResponse(
                success=True,
                message=f"Successfully indexed directory: {directory_path}",
                num_documents=result.get("num_documents"),
                details=result,
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error"))

    except Exception as e:
        logger.error(f"Index directory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_knowledge_base():
    """
    Clear the entire knowledge base.
    """
    registry = get_registry()

    try:
        success = registry.get("vector_store").delete_collection()
        if success:
            return {"success": True, "message": "Knowledge base cleared"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear knowledge base")
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_knowledge_base_stats():
    """
    Get knowledge base statistics.
    """
    from app.services.knowledge_base_service import get_knowledge_base_service

    registry = get_registry()
    kb_service = get_knowledge_base_service()

    try:
        kb_stats = kb_service.get_stats()
        vector_stats = registry.get("vector_store").get_collection_stats()

        return {
            "success": True,
            "knowledge_base": kb_stats,
            "vector_store": vector_stats,
        }
    except Exception as e:
        logger.error(f"Failed to get knowledge base stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reindex")
async def reindex_knowledge_base():
    """
    Re-index all documents in the knowledge base directory.
    """
    from app.services.knowledge_base_service import get_knowledge_base_service

    registry = get_registry()
    kb_service = get_knowledge_base_service()

    try:
        kb_service._is_initialized = False
        kb_service._indexed_files.clear()

        result = await kb_service.initialize(registry.get("rag"))

        return {
            "success": result.get("success", False),
            "message": "Knowledge base re-indexed" if result.get("success") else "Re-indexing failed",
            "details": result,
        }
    except Exception as e:
        logger.error(f"Failed to reindex knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diagnostics")
async def get_retrieval_diagnostics(test_query: Optional[str] = Query(None, description="可选的测试查询")):
    """
    获取检索系统诊断信息。

    检查各个组件的状态，包括:
    - 知识库索引状态
    - BM25 索引状态
    - 向量存储状态
    - 重排序器状态

    可选传入测试查询来验证检索功能。
    """
    from app.services.knowledge_base_service import get_knowledge_base_service

    kb_service = get_knowledge_base_service()

    try:
        diagnostics = await kb_service.get_retrieval_diagnostics(test_query)
        return {
            "success": True,
            **diagnostics,
        }
    except Exception as e:
        logger.error(f"Failed to get retrieval diagnostics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rebuild-bm25")
async def rebuild_bm25_index():
    """
    手动重建 BM25 索引。

    当检索结果不理想时，可以使用此接口重建 BM25 索引，
    以确保混合检索正常工作。
    """
    from app.services.knowledge_base_service import get_knowledge_base_service

    kb_service = get_knowledge_base_service()

    try:
        result = await kb_service._rebuild_bm25_index()
        return {
            "success": result.get("success", False),
            "message": "BM25 index rebuilt" if result.get("success") else "BM25 rebuild failed",
            "details": result,
        }
    except Exception as e:
        logger.error(f"Failed to rebuild BM25 index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Batch Upload Endpoints ==============

@router.post("/upload/batch", response_model=BatchUploadResponse)
async def create_batch_upload(request: BatchUploadRequest):
    """
    Create a batch upload task.
    """
    registry = get_registry()

    try:
        task = registry.get("upload_service").create_batch_task(request.files)

        return BatchUploadResponse(
            task_id=task.id,
            total_files=task.total_files,
            files=[
                {
                    "id": f.id,
                    "filename": f.filename,
                    "status": f.status.value,
                    "error": f.error_message,
                }
                for f in task.files
            ],
        )
    except Exception as e:
        logger.error(f"Batch upload creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/batch/{task_id}/file/{file_id}")
async def upload_batch_file(
    task_id: str,
    file_id: str,
    file: UploadFile = File(...),
):
    """
    Upload a single file within a batch task.
    """
    registry = get_registry()

    task = registry.get("upload_service").get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    file_info = None
    for f in task.files:
        if f.id == file_id:
            file_info = f
            break

    if not file_info:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found in task")

    try:
        content = await file.read()
        result = await registry.get("upload_service").process_file(task_id, file_id, content)

        return {
            "file_id": result.id,
            "filename": result.filename,
            "status": result.status.value,
            "progress": result.progress,
            "chunks": result.chunks_processed,
            "error": result.error_message,
        }
    except Exception as e:
        logger.error(f"Batch file upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upload/batch/{task_id}/status")
async def get_batch_upload_status(task_id: str):
    """
    Get the status of a batch upload task.
    """
    registry = get_registry()

    status = registry.get("upload_service").get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return status


@router.post("/upload/batch/{task_id}/stream")
async def stream_batch_upload(
    task_id: str,
    files: List[UploadFile] = File(...),
):
    """
    Upload multiple files and stream progress updates.
    """
    registry = get_registry()

    task = registry.get("upload_service").get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    file_contents = []
    for i, upload_file in enumerate(files):
        if i < len(task.files):
            content = await upload_file.read()
            file_contents.append((task.files[i].id, content))

    async def generate():
        async for event in registry.get("upload_service").process_batch(task_id, file_contents):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


@router.post("/upload/multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple files at once (simplified batch upload).
    """
    registry = get_registry()

    file_infos = [
        {"filename": f.filename or "unknown", "file_size": f.size or 0}
        for f in files
    ]

    task = registry.get("upload_service").create_batch_task(file_infos)

    results = []
    for i, upload_file in enumerate(files):
        if i >= len(task.files):
            break

        file_info = task.files[i]
        try:
            content = await upload_file.read()
            result = await registry.get("upload_service").process_file(
                task.id, file_info.id, content
            )
            results.append({
                "filename": result.filename,
                "status": result.status.value,
                "chunks": result.chunks_processed,
                "error": result.error_message,
            })
        except Exception as e:
            results.append({
                "filename": file_info.filename,
                "status": "failed",
                "chunks": 0,
                "error": str(e),
            })

    return {
        "task_id": task.id,
        "total_files": task.total_files,
        "completed_files": task.completed_files,
        "failed_files": task.failed_files,
        "results": results,
    }


# ============== Hybrid Search & ES Endpoints ==============

@router.post("/search-advanced", response_model=AdvancedSearchResponse)
async def search_knowledge_advanced(request: AdvancedSearchRequest):
    """
    Advanced search with hybrid retrieval (ES BM25 + Milvus vector).

    Features:
    - Combines keyword matching (BM25) and semantic search (vectors)
    - Supports metadata filtering (file_type, source, date ranges, etc.)
    - Configurable weights for BM25 and vector components
    - Uses RRF (Reciprocal Rank Fusion) for result merging

    Falls back to vector-only search if ES is not available.
    """
    registry = get_registry()

    try:
        # Check if hybrid search is available
        hybrid_retriever = registry.get("hybrid_retriever")

        if hybrid_retriever:
            # Use hybrid retriever
            docs = await hybrid_retriever._aget_relevant_documents(request.query)

            results = [
                AdvancedSearchResult(
                    chunk_id=doc.metadata.get("chunk_id", ""),
                    content=doc.page_content,
                    score=doc.metadata.get("score", 0.0),
                    source=doc.metadata.get("retrieval_source", "hybrid"),
                    metadata=doc.metadata if request.include_metadata else {},
                )
                for doc in docs[:request.top_k]
            ]

            return AdvancedSearchResponse(
                query=request.query,
                results=results,
                total_count=len(results),
                search_type="hybrid",
                weights={
                    "vector": request.vector_weight,
                    "bm25": request.bm25_weight,
                },
            )

        else:
            # Fall back to vector-only search
            rag = registry.get("rag")
            vector_results = rag.get_relevant_documents(request.query)

            results = [
                AdvancedSearchResult(
                    chunk_id=r.get("id", ""),
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                    source="milvus",
                    metadata=r.get("metadata", {}) if request.include_metadata else {},
                )
                for r in vector_results[:request.top_k]
            ]

            return AdvancedSearchResponse(
                query=request.query,
                results=results,
                total_count=len(results),
                search_type="vector_only",
                weights={"vector": 1.0},
            )

    except Exception as e:
        logger.error(f"Advanced search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage/health")
async def get_storage_health():
    """
    Get health status of all storage backends (Milvus + Elasticsearch).

    Returns connection status and basic stats for each storage system.
    """
    registry = get_registry()

    try:
        health = {
            "status": "healthy",
            "milvus": {"status": "unknown"},
            "elasticsearch": {"status": "unavailable"},
            "hybrid_enabled": settings.HYBRID_STORAGE_ENABLED,
        }

        # Check Milvus
        vector_store = registry.get("vector_store")
        if vector_store:
            try:
                stats = vector_store.get_collection_stats()
                if "error" not in stats:
                    health["milvus"] = {
                        "status": "healthy",
                        "collection": stats.get("name"),
                        "num_entities": stats.get("num_entities", 0),
                    }
                else:
                    health["milvus"] = {"status": "error", "error": stats.get("error")}
            except Exception as e:
                health["milvus"] = {"status": "error", "error": str(e)}

        # Check Elasticsearch
        es_manager = registry.get("es_manager")
        if es_manager:
            try:
                es_health = await es_manager.health_check()
                health["elasticsearch"] = es_health
            except Exception as e:
                health["elasticsearch"] = {"status": "error", "error": str(e)}

        # Check hybrid store
        hybrid_store = registry.get("hybrid_store")
        if hybrid_store:
            try:
                hybrid_health = await hybrid_store.health_check()
                health["hybrid_store"] = hybrid_health
            except Exception as e:
                health["hybrid_store"] = {"status": "error", "error": str(e)}

        # Overall status
        milvus_ok = health["milvus"].get("status") == "healthy"
        es_ok = health["elasticsearch"].get("status") == "healthy"

        if not milvus_ok:
            health["status"] = "degraded"
        if settings.HYBRID_STORAGE_ENABLED and not es_ok:
            health["status"] = "degraded"

        return health

    except Exception as e:
        logger.error(f"Storage health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage/stats")
async def get_storage_stats():
    """
    Get detailed statistics from all storage backends.

    Returns document counts, index sizes, and configuration info.
    """
    registry = get_registry()

    try:
        stats = {
            "milvus": {},
            "elasticsearch": {},
            "hybrid_storage": {
                "enabled": settings.HYBRID_STORAGE_ENABLED,
                "weights": {
                    "vector": settings.HYBRID_MILVUS_WEIGHT,
                    "bm25": settings.HYBRID_ES_WEIGHT,
                },
            },
        }

        # Milvus stats
        vector_store = registry.get("vector_store")
        if vector_store:
            try:
                stats["milvus"] = vector_store.get_collection_stats()
            except Exception as e:
                stats["milvus"] = {"error": str(e)}

        # Elasticsearch stats
        es_manager = registry.get("es_manager")
        if es_manager:
            try:
                stats["elasticsearch"] = await es_manager.get_stats()
            except Exception as e:
                stats["elasticsearch"] = {"error": str(e)}

        # Hybrid store stats
        hybrid_store = registry.get("hybrid_store")
        if hybrid_store:
            try:
                stats["hybrid_store"]["stats"] = await hybrid_store.get_stats()
            except Exception as e:
                stats["hybrid_store"]["error"] = str(e)

        return {"success": True, **stats}

    except Exception as e:
        logger.error(f"Storage stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/document/{doc_id}")
async def get_document_info(doc_id: str):
    """
    Get detailed information about a document by ID.

    Returns all chunks, metadata, and storage status.
    Requires Elasticsearch to be enabled.
    """
    registry = get_registry()

    try:
        # Try hybrid store first
        hybrid_store = registry.get("hybrid_store")
        if hybrid_store:
            info = await hybrid_store.get_document_info(doc_id)
            if info:
                return {"success": True, **info}

        # Try ES manager directly
        es_manager = registry.get("es_manager")
        if es_manager:
            metadata = await es_manager.get_document_metadata(doc_id)
            if metadata:
                return {
                    "success": True,
                    "doc_id": doc_id,
                    "metadata": metadata,
                    "source": "elasticsearch",
                }

        raise HTTPException(
            status_code=404,
            detail=f"Document {doc_id} not found or ES not enabled"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document from all storage backends.

    Removes chunks from both Elasticsearch and Milvus.
    """
    registry = get_registry()

    try:
        result = {"doc_id": doc_id, "deleted": False}

        # Use hybrid store for coordinated deletion
        hybrid_store = registry.get("hybrid_store")
        if hybrid_store:
            deletion_result = await hybrid_store.delete_document(doc_id)
            return {
                "success": deletion_result.get("success", False),
                **deletion_result,
            }

        # Fall back to ES-only deletion
        es_manager = registry.get("es_manager")
        if es_manager:
            deleted_count = await es_manager.delete_by_doc_id(doc_id)
            return {
                "success": deleted_count > 0,
                "doc_id": doc_id,
                "es_deleted": deleted_count,
            }

        raise HTTPException(
            status_code=400,
            detail="Document deletion requires Elasticsearch to be enabled"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== RAG Quality Evaluation (LlamaIndex + RAGAS) ==============

class RAGEvaluationRequest(BaseModel):
    """RAG 质量评估请求"""
    question: str = Field(..., description="用户问题")
    reference: Optional[str] = Field(None, description="参考答案（可选，用于正确性评估）")
    top_k: int = Field(default=5, ge=1, le=20, description="检索结果数量")
    evaluate_retrieval: bool = Field(default=True, description="是否评估检索质量")
    evaluate_generation: bool = Field(default=True, description="是否评估生成质量")


class RAGEvaluationResponse(BaseModel):
    """RAG 评估结果"""
    query: str
    response: str

    # 检索质量指标 (RAGAS)
    context_precision: float = Field(default=0.0, description="上下文精确度 (0-1, RAGAS)")
    context_recall: float = Field(default=0.0, description="上下文召回率 (0-1, RAGAS)")

    # 生成质量指标 (LlamaIndex)
    faithfulness: float = Field(default=0.0, description="答案忠实度 (0-1, LlamaIndex)")
    relevancy: float = Field(default=0.0, description="答案相关性 (0-1, LlamaIndex)")
    correctness: float = Field(default=0.0, description="答案正确性 (0-1, LlamaIndex)")

    # 综合评分
    overall_score: float = Field(..., description="综合评分 (0-1)")

    # 元数据
    evaluator_type: str = Field(..., description="评估器类型 (llamaindex/ragas/hybrid)")
    evaluation_time_ms: int = Field(..., description="评估耗时(ms)")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")


@router.post("/rag/ragas-evaluate", response_model=RAGEvaluationResponse)
async def evaluate_rag_quality(request: RAGEvaluationRequest):
    """
    使用 LlamaIndex Evaluation + RAGAS 框架进行 RAG 质量评估。

    **LlamaIndex Evaluators**:
    - FaithfulnessEvaluator: 评估答案是否忠实于上下文
    - RelevancyEvaluator: 评估答案与问题的相关性
    - CorrectnessEvaluator: 评估答案的正确性（需要 reference）

    **RAGAS Metrics** (如果已安装):
    - context_precision: 上下文精确度
    - context_recall: 上下文召回率
    - faithfulness: 忠实度
    - answer_relevancy: 答案相关性

    注意: 此评估会调用 LLM 进行评判，可能需要几秒钟完成。
    """
    from app.rag.ragas_evaluator import get_rag_evaluator
    from app.rag.query_engine import get_query_engine

    try:
        # 获取 RAG 查询引擎和评估器
        query_engine = get_query_engine()
        evaluator = get_rag_evaluator()

        # 执行 RAG 查询
        rag_response = await query_engine.aquery(request.question)

        # 提取上下文和答案
        contexts = [s.get("content", "") for s in rag_response.sources]
        answer = rag_response.answer

        # 执行评估
        result = await evaluator.evaluate(
            query=request.question,
            response=answer,
            contexts=contexts,
            reference=request.reference,
            evaluate_retrieval=request.evaluate_retrieval,
            evaluate_generation=request.evaluate_generation,
        )

        return RAGEvaluationResponse(
            query=result.query,
            response=answer[:500] + "..." if len(answer) > 500 else answer,
            context_precision=result.retrieval.context_precision,
            context_recall=result.retrieval.context_recall,
            faithfulness=result.generation.faithfulness,
            relevancy=max(result.generation.relevancy, result.generation.answer_relevancy),
            correctness=result.generation.correctness,
            overall_score=result.overall_score,
            evaluator_type=result.evaluator_type,
            evaluation_time_ms=result.evaluation_time_ms,
            suggestions=result.suggestions,
        )

    except Exception as e:
        logger.error(f"RAG evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class RAGBatchEvaluationRequest(BaseModel):
    """RAG 批量评估请求"""
    test_cases: List[Dict[str, Any]] = Field(
        ...,
        description="测试用例列表，每个包含 question, reference(可选)",
        examples=[[
            {"question": "支付方式有哪些？", "reference": "支持微信、支付宝、银行卡"},
            {"question": "如何退款？"},
        ]]
    )
    max_concurrent: int = Field(default=3, ge=1, le=10, description="最大并发数")


@router.post("/rag/ragas-batch-evaluate")
async def batch_evaluate_rag(request: RAGBatchEvaluationRequest):
    """
    批量执行 RAG 质量评估。

    使用 LlamaIndex Evaluation + RAGAS 评估多个测试用例。

    适用于：
    - RAG 系统的回归测试
    - 不同配置的对比评估
    - 知识库质量检测

    注意: 批量评估可能需要较长时间，建议控制测试用例数量。
    """
    from app.rag.ragas_evaluator import get_rag_evaluator
    from app.rag.query_engine import get_query_engine

    try:
        query_engine = get_query_engine()
        evaluator = get_rag_evaluator()

        results = []
        for case in request.test_cases:
            try:
                question = case.get("question", "")

                # 执行 RAG 查询
                rag_response = await query_engine.aquery(question)

                # 提取上下文和答案
                contexts = [s.get("content", "") for s in rag_response.sources]
                answer = rag_response.answer

                # 执行评估
                result = await evaluator.evaluate(
                    query=question,
                    response=answer,
                    contexts=contexts,
                    reference=case.get("reference"),
                )

                results.append({
                    "question": question,
                    "status": "success",
                    "evaluator_type": result.evaluator_type,
                    "overall_score": result.overall_score,
                    "retrieval": {
                        "context_precision": result.retrieval.context_precision,
                        "context_recall": result.retrieval.context_recall,
                    },
                    "generation": {
                        "faithfulness": result.generation.faithfulness,
                        "relevancy": max(result.generation.relevancy, result.generation.answer_relevancy),
                        "correctness": result.generation.correctness,
                    },
                    "suggestions": result.suggestions,
                })

            except Exception as e:
                results.append({
                    "question": case.get("question", ""),
                    "status": "error",
                    "error": str(e),
                })

        # 计算聚合指标
        successful_results = [r for r in results if r.get("status") == "success"]

        if successful_results:
            def avg(key_path: str) -> float:
                values = []
                for r in successful_results:
                    value = r
                    for key in key_path.split("."):
                        value = value.get(key, 0) if isinstance(value, dict) else 0
                    if value:
                        values.append(value)
                return round(sum(values) / len(values), 4) if values else 0.0

            aggregated = {
                "total_cases": len(request.test_cases),
                "successful_cases": len(successful_results),
                "failed_cases": len(results) - len(successful_results),
                "avg_overall_score": avg("overall_score"),
                "retrieval": {
                    "avg_context_precision": avg("retrieval.context_precision"),
                    "avg_context_recall": avg("retrieval.context_recall"),
                },
                "generation": {
                    "avg_faithfulness": avg("generation.faithfulness"),
                    "avg_relevancy": avg("generation.relevancy"),
                    "avg_correctness": avg("generation.correctness"),
                },
            }
        else:
            aggregated = {
                "total_cases": len(request.test_cases),
                "successful_cases": 0,
                "failed_cases": len(results),
            }

        return {
            "success": True,
            "aggregated_metrics": aggregated,
            "detailed_results": results,
        }

    except Exception as e:
        logger.error(f"Batch RAG evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/ragas-stats")
async def get_ragas_stats():
    """
    获取 RAGAS 评估的聚合统计信息。

    返回自上次重置以来所有评估的汇总指标，包括:
    - 平均检索质量指标
    - 平均生成质量指标
    - 幻觉检测率
    """
    from app.rag.ragas_evaluator import get_ragas_evaluator

    try:
        evaluator = get_ragas_evaluator()
        metrics = evaluator.get_aggregated_metrics()

        if not metrics:
            return {
                "success": True,
                "message": "暂无 RAGAS 评估记录",
                "metrics": {},
            }

        return {
            "success": True,
            "metrics": metrics,
        }

    except Exception as e:
        logger.error(f"Get RAGAS stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/ragas-history")
async def get_ragas_history(limit: int = Query(default=20, ge=1, le=100)):
    """
    获取 RAGAS 评估历史记录。

    返回最近的评估详细结果。
    """
    from app.rag.ragas_evaluator import get_ragas_evaluator

    try:
        evaluator = get_ragas_evaluator()
        history = evaluator.get_evaluation_history(limit=limit)

        return {
            "success": True,
            "count": len(history),
            "history": history,
        }

    except Exception as e:
        logger.error(f"Get RAGAS history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/ragas-reset")
async def reset_ragas_evaluator():
    """
    重置 RAGAS 评估器，清除所有历史记录和统计。
    """
    from app.rag.ragas_evaluator import get_ragas_evaluator

    try:
        evaluator = get_ragas_evaluator()
        evaluator.reset()

        return {
            "success": True,
            "message": "RAGAS 评估器已重置",
        }

    except Exception as e:
        logger.error(f"Reset RAGAS evaluator error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Pipeline Stage Quality Monitoring ==============

@router.get("/rag/pipeline-stats")
async def get_pipeline_stage_stats():
    """
    获取 RAG 流水线各阶段的质量统计。

    返回各阶段的平均指标:
    - 查询转换: 扩展查询数量
    - 检索: 平均检索数量和分数
    - 重排序: 平均重排序数量和分数
    - 后处理: 平均最终文档数量
    - 质量分布: excellent/good/fair/poor 的分布
    """
    from app.rag.ragas_evaluator import get_pipeline_monitor

    try:
        monitor = get_pipeline_monitor()
        stats = monitor.get_stage_stats()

        if not stats:
            return {
                "success": True,
                "message": "暂无流水线监控记录",
                "stats": {},
            }

        return {
            "success": True,
            "stats": stats,
        }

    except Exception as e:
        logger.error(f"Get pipeline stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/pipeline-history")
async def get_pipeline_stage_history(limit: int = Query(default=20, ge=1, le=100)):
    """
    获取 RAG 流水线各阶段的质量历史记录。

    返回最近查询的各阶段详细指标。
    """
    from app.rag.ragas_evaluator import get_pipeline_monitor

    try:
        monitor = get_pipeline_monitor()
        history = monitor.get_stage_history(limit=limit)

        return {
            "success": True,
            "count": len(history),
            "history": history,
        }

    except Exception as e:
        logger.error(f"Get pipeline history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/pipeline-reset")
async def reset_pipeline_monitor():
    """
    重置流水线质量监控器。
    """
    from app.rag.ragas_evaluator import get_pipeline_monitor

    try:
        monitor = get_pipeline_monitor()
        monitor.reset()

        return {
            "success": True,
            "message": "流水线质量监控器已重置",
        }

    except Exception as e:
        logger.error(f"Reset pipeline monitor error: {e}")
        raise HTTPException(status_code=500, detail=str(e))