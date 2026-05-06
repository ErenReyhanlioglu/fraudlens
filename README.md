---
title: FraudLens
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.44.0"
app_file: demo/app_hf.py
pinned: false
---

## Project Structure

```text
src/fraudlens/
├── api/      # FastAPI routes, middleware
├── agents/   # LangGraph agents + tools 
├── core/     # config.py (Pydantic Settings), logging, exceptions
├── db/       # SQLAlchemy models, session, Alembic migrations
├── llm/      # LLM provider routing 
├── ml/       # XGBoost, SHAP, feature_extractor, model serving
├── rag/      # Qdrant, chunker, embedder, retriever 
└── schemas/  # Pydantic models: transaction, decision, investigation, sar
The project follows a standard structure for modern Python applications, separating concerns into distinct modules.

# Manage infrastructure
docker compose up -d
docker compose down
```
src/fraudlens/
├── api/       # FastAPI routes, middleware
├── agents/    # LangGraph agents + tools
├── core/      # config.py (Pydantic Settings), logging, exceptions
├── db/        # SQLAlchemy models, session, Alembic migrations
├── ml/        # XGBoost, SHAP, feature_extractor, model serving
├── rag/       # Qdrant, chunker, embedder, retriever
└── schemas/   # Pydantic models for data validation
```

## Quick Start & Usage

### 1. Setup and Run

First, start the required services (PostgreSQL, Redis, Qdrant):
```bash
docker compose up -d
```

# Query historical decisions (default: last 10)
uv run python scripts/investigator_agent_history.py
uv run python scripts/investigator_agent_history.py --limit 5
uv run python scripts/investigator_agent_history.py --hint suspicious
uv run python scripts/investigator_agent_history.py --since 24h
uv run python scripts/investigator_agent_history.py --since 7d --verbose
Then, apply database migrations:
```bash
uv run alembic upgrade head
```

# Run health checks
uv run python scripts/critical_agent_healthcheck.py
Finally, start the FastAPI application:
```bash
uv run uvicorn src.fraudlens.api.main:app --reload --port 8001
```

# Query historical decisions (default: last 10)
uv run python scripts/critical_agent_history.py
uv run python scripts/critical_agent_history.py --limit 5
uv run python scripts/critical_agent_history.py --hint suspicious
uv run python scripts/critical_agent_history.py --since 24h
uv run python scripts/critical_agent_history.py --verbose --limit 3
The API will be available at `http://localhost:8001`.

# Run health checks with specific configurations
uv run scripts/sar_agent_healthcheck.py
uv run scripts/sar_agent_healthcheck.py --no-langsmith
uv run scripts/sar_agent_healthcheck.py --port 8001 --host 127.0.0.1
uv run scripts/sar_agent_healthcheck.py --seed 123
### 2. Running the Evaluation

# Query historical decisions (default: last 10)
uv run scripts/sar_agent_history.py 
uv run scripts/sar_agent_history.py --limit 5
uv run scripts/sar_agent_history.py --since 24h
uv run scripts/sar_agent_history.py --since 7d --verbose
To run the gold-standard evaluation set against the running API:
```bash
# Ensure the API is running on localhost:8001 before starting
uv run tests/eval/run_eval.py --concurrency 3
```
Results will be saved to `tests/eval/results/`.

http://localhost:8001/docs          FastAPI Swagger UI Interactive API endpoint documentation
http://localhost:8001/health        API Health Check System status verification
http://localhost:5000               MLflow UI Experiment tracking and model registry
http://localhost:6333/dashboard     Qdrant Dashboard Vector database visualization
localhost:5432                      PostgreSQL DatabasePostgreSQL Relational database
localhost:6379                      Redis In-memory cache (connect via CLI)
### 3. Running the Demo

# Run the linter on all Python files in the current directory
ruff check .
To launch the Streamlit demo application:
```bash
uv run streamlit run demo/app.py
```

# Run the linter on a specific file or directory
ruff check path/to/code/
### 4. Development & Linting

# Run the linter and automatically apply safe fixes
ruff check . --fix
This project uses `ruff` for linting and formatting.
```bash
# Check for linting and formatting issues
uv run ruff check .
uv run ruff format --check .

# Run the linter and apply all fixes, including potentially unsafe ones
ruff check . --fix --unsafe-fixes
# Automatically fix issues
uv run ruff format .
uv run ruff check . --fix
```

uv run tests/eval/run_eval.py                   # eval runner
uv run --group demo streamlit run demo/app.py   # Streamlit demo
### 5. Utility Scripts

```
The project includes several scripts for interacting with the system.
```bash
# Build and populate the RAG index in Qdrant