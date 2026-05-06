"""Tool: regulatory policy lookup via RAG over FATF/MASAK documents."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from fraudlens.rag.retriever import retrieve


@tool
async def regulatory_policy_rag(query: str) -> str:
    """Look up AML/CFT regulatory requirements from FATF and MASAK guidance documents.

    Use this tool to check what regulations say about a specific risk indicator,
    transaction type, or compliance requirement. Always cite the source in your reasoning.

    Args:
        query: A specific regulatory question or topic, e.g.
               "FATF requirements for high-value cash transactions" or
               "MASAK suspicious transaction reporting thresholds".

    Returns:
        JSON with a list of relevant excerpts, each containing the text,
        source document name, page number, and similarity score.
        Includes a formatted citation string for each excerpt.
    """
    bddk_query = f"BDDK MASAK şüpheli işlem {query}"
    chunks_a, chunks_b = await retrieve(query, top_k=5), await retrieve(bddk_query, top_k=5)

    # Merge by (source, page), keep highest score per unique chunk
    seen: dict[tuple[str, int], dict] = {}
    for chunk in chunks_a + chunks_b:
        key = (chunk["source"], chunk["page"])
        if key not in seen or chunk["score"] > seen[key]["score"]:
            seen[key] = chunk

    merged = sorted(seen.values(), key=lambda c: c["score"], reverse=True)[:5]

    if not merged:
        return json.dumps(
            {
                "query": query,
                "status": "no_results",
                "message": "RAG index may not be built yet or no relevant content found.",
                "excerpts": [],
            }
        )

    excerpts = []
    for chunk in merged:
        excerpts.append(
            {
                "text": chunk["text"],
                "citation": f"{chunk['source']}, p.{chunk['page']}",
                "source": chunk["source"],
                "page": chunk["page"],
                "relevance_score": round(chunk["score"], 4),
            }
        )

    return json.dumps({"query": query, "excerpts": excerpts}, ensure_ascii=False)
