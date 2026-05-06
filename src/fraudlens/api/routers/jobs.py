"""GET /jobs/{job_id} endpoint for polling background pipeline status."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fraudlens.core.job_store import get_job

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["jobs"])


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage_label: str
    current_tool: str | None
    completed_tools: list[str]
    thought: str | None
    result: dict | None
    error: str | None
    updated_at: str


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Return the current state of a background fraud investigation job."""
    state = await get_job(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(
        job_id=state["job_id"],
        status=state["status"],
        stage_label=state["stage_label"],
        current_tool=state.get("current_tool"),
        completed_tools=state.get("completed_tools") or [],
        thought=state.get("thought"),
        result=state.get("result"),
        error=state.get("error"),
        updated_at=state["updated_at"],
    )
