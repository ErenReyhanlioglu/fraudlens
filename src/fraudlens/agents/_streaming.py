"""Shared streaming utilities for LangGraph ReAct agents.

Handles astream_events processing, Redis job state updates,
and message history extraction from the event stream.
"""

from __future__ import annotations

from typing import Any

import structlog

from fraudlens.core.job_store import get_job, update_job
from fraudlens.core.llm_throttle import throttled_stream

logger = structlog.get_logger(__name__)


async def safe_update_job(job_id: str, **kwargs: Any) -> None:
    """Update Redis job state; swallows all errors so the pipeline never stops."""
    if not job_id:
        return
    try:
        await update_job(job_id, **kwargs)
    except Exception as exc:
        logger.warning("job_update_failed", job_id=job_id, error=str(exc))


async def get_completed_tools(job_id: str) -> list[str]:
    """Return the current completed_tools list from Redis."""
    if not job_id:
        return []
    try:
        state = await get_job(job_id)
        return state.get("completed_tools", []) if state else []
    except Exception:
        return []


async def stream_agent_events(
    agent: Any,
    input_messages: dict[str, Any],
    job_id: str,
) -> tuple[list[Any], str]:
    """Run a LangGraph ReAct agent via astream_events and update Redis on each event.

    Event handling:
    - on_chat_model_stream : accumulate thought; flush to Redis every ~100 chars
    - on_tool_start        : set current_tool, reset thought buffer
    - on_tool_end          : append to completed_tools, clear current_tool
    - on_chain_end         : capture messages when output contains them

    Args:
        agent: A compiled LangGraph graph (create_react_agent or similar).
        input_messages: Dict passed to astream_events (e.g. {"messages": [...]}).
        job_id: Redis job identifier; pass "" to disable all Redis updates.

    Returns:
        messages: Full LangGraph message history extracted from the last chain-end
                  event that carries a "messages" key in its output.
        final_thought: Complete accumulated LLM text from the run.
    """
    thought_buffer = ""
    last_written_len = 0
    messages: list[Any] = []

    async with throttled_stream("agent_stream"):
        async for event in agent.astream_events(input_messages, version="v2"):
            kind: str = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content"):
                    content = chunk.content
                    if isinstance(content, str):
                        thought_buffer += content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                thought_buffer += block.get("text", "")

                    if job_id and len(thought_buffer) - last_written_len >= 100:
                        await safe_update_job(job_id, thought=thought_buffer.strip())
                        last_written_len = len(thought_buffer)

            elif kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                thought_buffer = ""
                last_written_len = 0
                await safe_update_job(
                    job_id,
                    current_tool=tool_name,
                    thought=f"Calling {tool_name}...",
                )

            elif kind == "on_tool_end":
                tool_name = event.get("name", "unknown")
                completed = await get_completed_tools(job_id)
                if tool_name not in completed:
                    completed = [*completed, tool_name]
                await safe_update_job(
                    job_id,
                    current_tool=None,
                    completed_tools=completed,
                )

            elif kind == "on_chain_end":
                output = event["data"].get("output", {})
                if isinstance(output, dict) and "messages" in output:
                    messages = output["messages"]

    if job_id and thought_buffer.strip():
        await safe_update_job(job_id, thought=thought_buffer.strip())

    return messages, thought_buffer
