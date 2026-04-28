# FraudLens

Multi-agent fraud/AML detection system. XGBoost scores transactions, LangGraph agents investigate suspicious ones, RAG grounds decisions in BDDK/FATF regulations.

## Terminal

docker compose up -d
docker compose down
uv run uvicorn src.fraudlens.api.main:app --reload --port 8001
uv run python scripts/build_rag_index.py
uv run python scripts/investigator_agent_healthcheck.py
uv run python scripts/critical_agent_healthcheck.py

# Past Investigation Agent decisions (default: last 10)
uv run python scripts/investigator_agent_history.py
uv run python scripts/investigator_agent_history.py --limit 5
uv run python scripts/investigator_agent_history.py --hint suspicious
uv run python scripts/investigator_agent_history.py --since 24h
uv run python scripts/investigator_agent_history.py --since 7d --verbose

# Past Critical Agent decisions (default: last 10)
uv run python scripts/critical_agent_history.py
uv run python scripts/critical_agent_history.py --limit 5
uv run python scripts/critical_agent_history.py --hint suspicious
uv run python scripts/critical_agent_history.py --since 24h
uv run python scripts/critical_agent_history.py --verbose --limit 3

http://localhost:8001/docs           # FastAPI Swagger UI (API endpoints)
http://localhost:8001/health         # API health check
http://localhost:5000                # MLflow tracking UI
http://localhost:6333/dashboard      # Qdrant vector DB dashboard
http://localhost:5432                # PostgreSQL (with DB client)
http://localhost:6379                # Redis (with CLI)

## Architecture

POST /transactions → XGBoost scorer (<50ms) → Triage Router:
  p < 0.3  → auto-approve
  0.3-0.7  → Investigation Agent (claude-haiku-4-5, 5 tools, LangGraph)
  p >= 0.7 → Critical Agent (claude-haiku-4-5, 8 tools + RAG, LangGraph)
→ Decision Synthesizer (Pydantic structured output)
→ SAR Generator (only on "escalate")

## Current State (Hafta 3 complete)

- XGBoost tuned model: data/processed/xgb_tuned_v1.joblib (79 features, PR-AUC 0.4834)
- SHAP explainer integrated, top-10 contributors per prediction
- FastAPI: POST /api/v1/transactions, GET /api/v1/decisions/{id}
- raw_mode=true bypasses feature_extractor, accepts IEEE-CIS features directly
- PostgreSQL decisions table live (Alembic migrated)
- Integration tests: 18/18 pass across all triage buckets
- LangSmith + MLflow configured, Docker stack healthy

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

## Key Decisions

- Triage Router is rule-based, NOT AI
- XGBoost for tabular scoring — deterministic, auditable, 100x faster than LLM
- claude-haiku-4-5 for investigation (~30%) and critical (~10%)
- SHAP on every prediction → passed as agent context via explain_ml_score tool
- RAG: 512 tok chunks, 128 overlap, BM25+dense hybrid, bge-reranker, citation mandatory
- IEEE-CIS ~3.5% fraud rate → class_weight balanced, NOT SMOTE
- Feature extractor maps banking API fields → IEEE-CIS features via JSON rule files

## Code Rules

- src layout, type hints everywhere, no exceptions
- async/await for FastAPI + SQLAlchemy + httpx
- Pydantic v2 strict mode + retry on LLM output failure
- structlog JSON, never print()
- English in all code, docstrings, commits
- Google-style docstrings on public classes/functions
- Ruff enforced: built-in types (list/dict/tuple not typing.*), sorted imports, 100 char limit

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