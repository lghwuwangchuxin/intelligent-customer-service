"""Evaluation routes for API Gateway."""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


class EvaluationDataItem(BaseModel):
    """Single evaluation data item."""
    question: str
    answer: str
    contexts: List[str] = []
    ground_truth: Optional[str] = None


class SubmitEvaluationRequest(BaseModel):
    """Submit evaluation request."""
    data: List[EvaluationDataItem]
    metrics: Optional[List[str]] = None


@router.post("/submit")
async def submit_evaluation(request: Request, body: SubmitEvaluationRequest):
    """
    Submit an evaluation task.
    """
    client = request.app.state.evaluation_client

    if not client:
        raise HTTPException(status_code=503, detail="Evaluation service unavailable")

    try:
        result = await client.submit_evaluation(
            data=[item.dict() for item in body.data],
            metrics=body.metrics,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{task_id}")
async def get_result(request: Request, task_id: str):
    """
    Get evaluation result.
    """
    client = request.app.state.evaluation_client

    if not client:
        raise HTTPException(status_code=503, detail="Evaluation service unavailable")

    try:
        result = await client.get_result(task_id)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_evaluations(
    request: Request,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """
    List evaluation tasks.
    """
    client = request.app.state.evaluation_client

    if not client:
        raise HTTPException(status_code=503, detail="Evaluation service unavailable")

    try:
        result = await client.list_evaluations(
            status=status,
            page=page,
            page_size=page_size,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
