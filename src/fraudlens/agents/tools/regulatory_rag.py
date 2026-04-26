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
    chunks = await retrieve(query, top_k=5)

    if not chunks:
        return json.dumps(
            {
                "query": query,
                "status": "no_results",
                "message": "RAG index may not be built yet or no relevant content found.",
                "excerpts": [],
            }
        )

    excerpts = []
    for chunk in chunks:
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
