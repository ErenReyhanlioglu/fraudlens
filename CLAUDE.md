# FraudLens

Multi-agent fraud/AML detection system. XGBoost scores transactions, LangGraph agents investigate suspicious ones, RAG grounds decisions in BDDK/FATF regulations.

## Commands
```bash
.venv\Scripts\activate              # Windows venv
docker compose up -d                # Postgres, Redis, Qdrant, MLflow
uv add <package>                    # Install dependency
pytest tests/                       # Run tests
ruff check . && ruff format .       # Lint + format
mypy src/                           # Type check
uvicorn src.fraudlens.api.main:app --reload  # Start API
```

## Architecture
```
POST /transactions → XGBoost scorer (<50ms) → Triage Router:
  p < 0.3  → auto-approve
  0.3–0.7  → Investigation Agent (Haiku, 5 tools, LangGraph)
  p ≥ 0.7  → Critical Agent (Sonnet, 8 tools + RAG, LangGraph)
→ Decision Synthesizer (Pydantic structured output)
→ SAR Generator (only on "escalate")
All calls traced via LangSmith, models versioned in MLflow.
```

## Code rules
- src layout: all code under `src/fraudlens/`
- Type hints on every function, no exceptions
- async/await for FastAPI, SQLAlchemy, httpx
- Pydantic v2 for all schemas and LLM output validation — strict mode + retry on failure
- structlog with JSON output, never print()
- English in all code, docstrings, commits
- Google-style docstrings on public classes/functions
- Tests alongside implementation, don't skip

## Key architectural decisions
- Layer 2 (Triage Router) is deliberately rule-based, NOT AI — routing is a business rule
- XGBoost for tabular scoring, NOT LLM — deterministic, auditable, 100x faster
- Haiku for high-volume investigation (~30% of txns), Sonnet only for critical (~10%) — cost-conscious routing
- SHAP values computed on every XGBoost prediction, passed as context to agents
- RAG: recursive chunking (512 tok, 128 overlap), hybrid retrieval (BM25 + dense), bge-reranker, citation mandatory
- All LLM structured outputs validated with Pydantic strict + retry pattern

## Gotchas
- IEEE-CIS dataset has ~3.5% fraud rate — use scale_pos_weight, NOT SMOTE
- .env for secrets, never hardcode — see .env.example
- data/raw/ and data/processed/ are gitignored
- Docker services: postgres:5432, redis:6379, qdrant:6333, mlflow:5000

## Ruff lint kuralları (Python 3.11+)

- `from typing import Tuple, List, Dict` kullanma → direkt `tuple`, `list`, `dict` yaz
- f-string içinde placeholder yoksa `f` prefix koyma → `print("text")` yaz
- Import sırası: stdlib → third-party → local, aralarında boş satır
- Satır max 1000 karakter
- Kullanılmayan import ekleme
- Loop variable kullanılmıyorsa `_` ile başlat → `for _key, val in ...`
- Lint fix komutu: `uv run ruff check . --fix --unsafe-fixes`