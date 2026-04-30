# FraudLens

Multi-agent fraud/AML detection system. XGBoost scores transactions, LangGraph agents investigate suspicious ones, RAG grounds decisions in BDDK/FATF regulations.

## Architecture Context
- **Pipeline:** POST /transactions -> XGBoost (determines bucket) -> Triage Router -> Agent (Investigation/Critical via LangGraph) -> Synthesizer -> SAR Generator.
- **Triage:** Rule-based routing based on ML probability (`p < 0.3` approve, `0.3-0.7` Investigation Agent, `p >= 0.7` Critical Agent).
- **Agents:** Powered by `claude-haiku-4-5`. Tools are selected based on strict docstrings. RAG implementation requires mandatory citations.
- **Data:** Imbalanced dataset handling via class weights, not SMOTE. Features extracted to IEEE-CIS format via JSON rules (`raw_mode=true` bypasses extraction).
- **Explainability:** SHAP values passed to agents via tools for all predictions.

## Project Structure

src/fraudlens/
├── api/      # FastAPI routes, middleware
├── agents/   # LangGraph agents + tools 
├── core/     # config.py (Pydantic Settings), logging, exceptions
├── db/       # SQLAlchemy models, session, Alembic migrations
├── llm/      # LLM provider routing 
├── ml/       # XGBoost, SHAP, feature_extractor, model serving
├── rag/      # Qdrant, chunker, embedder, retriever 
└── schemas/  # Pydantic models: transaction, decision, investigation, sar


## Code Rules & Restrictions (STRICT)
- **Language:** Code, docstrings, commits MUST be in English. (Turkish is allowed ONLY in our chat communication if requested).
- **Format:** Use `ruff` standards. Max 100 chars, sorted imports, built-in types (e.g., `list`, not `typing.List`).
- **Typing & Validation:** Strict type hints everywhere. Use Pydantic v2 strict mode.
- **Async:** Use `async/await` for FastAPI, SQLAlchemy, and httpx.
- **Logging:** Use `structlog` (JSON format). NEVER use `print()`.
- **LLM Outputs:** When modifying code, return ONLY the git diff or the specific changed lines. DO NOT output the entire file unless explicitly asked. DO NOT generate inline comments explaining the code.
- **Architecture:** Maintain strict separation of concerns (API -> Logic -> DB).

## Services & Keys

- postgres:5432, redis:6379, qdrant:6333, mlflow:5000
- ANTHROPIC_API_KEY → claude-haiku-4-5
- LANGSMITH_API_KEY + LANGSMITH_TRACING=true → auto-traces all LLM calls to fraudlens project
- All secrets in .env, never hardcode

## Gotchas

- data/raw/ and data/processed/ gitignored
- raw_mode=true on POST /transactions → score_raw(), direct IEEE-CIS dict to model
- Tool docstrings critical — LLM reads them to decide when to call each tool
- LangGraph state must be TypedDict
- Mock tools (similar_patterns, regulatory_rag) 

# Output Constraints (Optimized Caveman Mode)

1. Zero conversational filler. Do not use greetings, pleasantries, apologies, or ethical disclaimers.
2. No boilerplate text. Do not explain what you are going to do before doing it.
3. Code strictly speaks for itself. Provide direct code blocks without introductory or concluding paragraphs.
4. If reasoning is mathematically or algorithmically required before writing code, use ultra-concise bullet points or pseudocode. 
5. Maximize signal-to-noise ratio.