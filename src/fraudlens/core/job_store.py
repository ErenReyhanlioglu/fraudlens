"""Async Redis-backed job state store for background pipeline tracking."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import redis.asyncio as aioredis
import structlog

from fraudlens.core.config import get_settings

logger = structlog.get_logger(__name__)

JOB_TTL_SECONDS = 3600


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def _get_redis() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.from_url(settings.redis_url, decode_responses=True)


async def create_job(job_id: str) -> None:
    """Write initial queued state for a new job."""
    state: dict[str, object] = {
        "job_id": job_id,
        "status": "queued",
        "stage_label": "Queued for processing",
        "current_tool": None,
        "completed_tools": [],
        "thought": None,
        "result": None,
        "error": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    try:
        r = await _get_redis()
        await r.set(f"job:{job_id}", json.dumps(state), ex=JOB_TTL_SECONDS)
        await r.aclose()
        logger.debug("job_created", job_id=job_id)
    except Exception as exc:
        logger.warning("job_store_create_failed", job_id=job_id, error=str(exc))


async def update_job(job_id: str, **kwargs: object) -> None:
    """Update one or more fields in the job state. updated_at is always set."""
    try:
        r = await _get_redis()
        raw = await r.get(f"job:{job_id}")
        if raw is None:
            logger.warning("job_store_update_missing", job_id=job_id)
            await r.aclose()
            return
        state: dict = json.loads(raw)
        state.update(kwargs)
        state["updated_at"] = _now_iso()
        await r.set(f"job:{job_id}", json.dumps(state), ex=JOB_TTL_SECONDS)
        await r.aclose()
        logger.debug("job_updated", job_id=job_id, fields=list(kwargs.keys()))
    except Exception as exc:
        logger.warning("job_store_update_failed", job_id=job_id, error=str(exc))


async def get_job(job_id: str) -> dict | None:
    """Return the current job state dict, or None if not found."""
    try:
        r = await _get_redis()
        raw = await r.get(f"job:{job_id}")
        await r.aclose()
        if raw is None:
            return None
        return dict(json.loads(raw))
    except Exception as exc:
        logger.warning("job_store_get_failed", job_id=job_id, error=str(exc))
        return None
