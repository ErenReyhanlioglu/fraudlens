"""Global LLM concurrency gate with rate-limit retry.

A module-level asyncio.Semaphore caps simultaneous LLM agent invocations so the
combined token load stays within Anthropic's 50 K input-tokens / minute limit.

On anthropic.RateLimitError the call is retried with exponential backoff
(30 s → 60 s → 120 s) before re-raising. The semaphore is held for the FULL
duration of every retry so that a backoff-waiting invocation does not free a
slot for a new call that would receive the same 429 rejection.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TypeVar

import anthropic
import structlog

logger = structlog.get_logger(__name__)

_semaphore: asyncio.Semaphore | None = None
_LLM_MAX_CONCURRENT = 2
_MAX_RETRIES = 3
_BASE_WAIT = 30.0  # seconds; doubles each attempt: 30 → 60 → 120

T = TypeVar("T")


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_LLM_MAX_CONCURRENT)
    return _semaphore


async def throttled_invoke(fn: Callable[[], Awaitable[T]], label: str = "llm") -> T:
    """Invoke *fn* under the global LLM semaphore with rate-limit retry.

    Args:
        fn: Zero-argument async factory returning the desired value.
        label: Log identifier for this call site (e.g. "critical_agent").

    Raises:
        anthropic.RateLimitError: If all retry attempts are exhausted.
        Any other exception raised by *fn* propagates immediately without retry.
    """
    sem = _get_semaphore()
    async with sem:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return await fn()
            except anthropic.RateLimitError:
                if attempt == _MAX_RETRIES:
                    logger.error(
                        "rate_limit_exhausted",
                        label=label,
                        total_attempts=_MAX_RETRIES + 1,
                    )
                    raise
                wait = min(_BASE_WAIT * (2.0**attempt), 120.0)
                logger.warning(
                    "rate_limit_retry",
                    label=label,
                    attempt=attempt + 1,
                    wait_seconds=wait,
                )
                await asyncio.sleep(wait)

    raise AssertionError("unreachable")  # satisfies type checker


@asynccontextmanager
async def throttled_stream(label: str = "llm") -> AsyncGenerator[None, None]:
    """Acquire the global LLM semaphore for the duration of a streaming call.

    Use this instead of throttled_invoke when calling astream_events, since the
    semaphore must be held for the full streaming duration, not just until the
    first token arrives.
    """
    sem = _get_semaphore()
    logger.debug("throttled_stream_wait", label=label)
    async with sem:
        logger.debug("throttled_stream_start", label=label)
        yield
    logger.debug("throttled_stream_done", label=label)
