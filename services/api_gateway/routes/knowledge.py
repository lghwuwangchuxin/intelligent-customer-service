"""Knowledge/RAG routes for API Gateway."""

from typing import Optional, Dict
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from pydantic import BaseModel

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


class RetrieveRequest(BaseModel):
    """Retrieve request."""
    query: str
    knowledge_base_id: Optional[str] = None
    top_k: int = 10
    enable_rerank: bool = True


class IndexRequest(BaseModel):
    """Index document request."""
    content: str
    knowledge_base_id: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


@router.post("/retrieve")
async def retrieve(request: Request, body: RetrieveRequest):
    """
    Retrieve documents from knowledge base.
    """
    client = request.app.state.rag_client

    if not client:
        raise HTTPException(status_code=503, detail="RAG service unavailable")

    try:
        result = await client.retrieve(
            query=body.query,
            knowledge_base_id=body.knowledge_base_id,
            top_k=body.top_k,
            enable_rerank=body.enable_rerank,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index")
async def index_document(request: Request, body: IndexRequest):
    """
    Index a document.
    """
    client = request.app.state.rag_client

    if not client:
        raise HTTPException(status_code=503, detail="RAG service unavailable")

    try:
        result = await client.index_document(
            content=body.content,
            knowledge_base_id=body.knowledge_base_id,
            metadata=body.metadata,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    knowledge_base_id: Optional[str] = None,
):
    """
    Upload and index a document file.
    """
    client = request.app.state.rag_client

    if not client:
        raise HTTPException(status_code=503, detail="RAG service unavailable")

    try:
        content = await file.read()
        text_content = content.decode("utf-8")

        result = await client.index_document(
            content=text_content,
            knowledge_base_id=knowledge_base_id,
            metadata={"filename": file.filename},
        )
        return result

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(request: Request, document_id: str):
    """
    Delete a document.
    """
    client = request.app.state.rag_client

    if not client:
        raise HTTPException(status_code=503, detail="RAG service unavailable")

    try:
        success = await client.delete_document(document_id)
        return {"success": success}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents(
    request: Request,
    knowledge_base_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """
    List documents.
    """
    client = request.app.state.rag_client

    if not client:
        raise HTTPException(status_code=503, detail="RAG service unavailable")

    try:
        result = await client.list_documents(
            knowledge_base_id=knowledge_base_id,
            page=page,
            page_size=page_size,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
