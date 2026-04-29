 
# FraudLens — Architecture

## System Overview

FraudLens is a 6-layer multi-agent fraud detection system. XGBoost scores transactions in <50ms, LangGraph agents investigate suspicious ones, RAG grounds decisions in BDDK/FATF regulations, and a SAR generator produces regulatory reports for escalated cases.

## Layer Architecture

Layer 0 — Entry Point
  POST /api/v1/transactions (FastAPI, async)
  Accepts: TransactionRequest (real-world banking format) or raw IEEE-CIS features (raw_mode=true)
  Returns: TransactionResponse with score, triage action, SHAP contributors, investigation result

Layer 1 — Anomaly Scorer
  Model: XGBoost (xgb_tuned_v1.joblib, 79 features, PR-AUC 0.4834)
  Inference: <50ms, deterministic, auditable
  Explainability: TreeSHAP — top-10 contributors per prediction, passed as agent context
  Feature pipeline: feature_extractor.py maps banking fields → IEEE-CIS features via JSON rule files
  raw_mode=true: bypasses feature_extractor, accepts IEEE-CIS dict directly (demo/test use)

Layer 2 — Triage Router (rule-based, NOT AI)
  p < 0.30  → auto-approve  (no LLM call)
  0.30–0.70 → Investigation Agent (Layer 3a)
  p >= 0.70 → Critical Agent (Layer 3b)
  Thresholds are business rules, calibrated to fraud rate ~3.5%

Layer 3a — Investigation Agent
  Model: gemini-2.0-flash (langchain-google-genai)
  Framework: LangGraph ReAct (create_react_agent)
  Volume: ~30% of all transactions
  Tools (5):
    explain_ml_score     — SHAP values injected at agent-creation time (factory pattern, no DB round-trip)
    get_customer_history — PostgreSQL mock, deterministic via hashlib seed
    check_merchant_reputation — mock risk score, chargeback rate, flags
    get_geolocation_context  — mock VPN/impossible-travel detection
    find_similar_patterns    — mock (Hafta 5: Qdrant vector search)
  Output: InvestigationResult (Pydantic strict, 2 retries, INCONCLUSIVE fallback)
    decision_hint: likely_legitimate | suspicious | inconclusive
    confidence: float [0,1]
    evidence: list[str]
    red_flags: list[str]
    tools_called: list[str]
    reasoning_summary: str

Layer 3b — Critical Agent
  Model: claude-haiku-4.5
  Framework: LangGraph ReAct
  Volume: ~10% of all transactions
  Tools: Layer 3a tools (5) + deep_network_analysis + regulatory_policy_rag + adverse_media_search
  Status: Hafta 5 — currently delegates to Investigation Agent
  RAG: BDDK/FATF PDFs → Qdrant, hybrid BM25+dense retrieval, bge-reranker, citation mandatory

Layer 4 — Decision Synthesizer
  Model: claude-haiku-4.5
  Input: InvestigationResult + ML score
  Output: FraudDecision (Pydantic strict)
    decision: approve | decline | escalate
    confidence, ml_score, agent_findings, key_risk_factors, regulatory_triggers, reasoning_trace_id
  Status: Hafta 6

Layer 5 — SAR Report Generator
  Model: claude-haiku-4.5
  Triggered only on decision=escalate
  Template-based prompt, fixed sections:
    Customer Information, Transaction Details, Suspicious Indicators,
    Investigation Summary, Regulatory Trigger References, Recommended Action
  Status: Hafta 6

Layer 6 — Observability
  LangSmith: every LLM call auto-traced (LANGSMITH_TRACING=true → fraudlens project)
  MLflow: model versioning, metrics, artifacts (localhost:5000)
  structlog: JSON structured logging, never print()
  Status: LangSmith active, MLflow configured, Grafana Hafta 7

## Data Flow (End-to-End)

POST /transactions
  → FraudScorer.score() or score_raw()
  → SHAP top-10 computed
  → Triage Router
    → approve: Decision saved to DB, response returned
    → investigate: run_fraud_investigation(graph.py) called
        → Investigation Agent (ReAct loop)
            → tool calls (1–4 tools depending on findings)
            → InvestigationResult parsed + validated
        → Decision saved to DB with agent findings
        → response returned
    → escalate: Critical Agent (currently delegates to Investigation Agent)
        → Hafta 5: RAG + network analysis active
        → Hafta 6: SAR generated

## Database Schema

Table: decisions
  id: UUID (PK)
  transaction_id: str
  ml_score: float
  triage_action: enum (approve/investigate/escalate)
  outcome: enum (approve/decline/escalate)
  confidence: float
  reasoning: str
  shap_values: JSONB
  agent_findings: JSONB
  regulatory_citations: JSONB
  processing_time_ms: int
  created_at: timestamp

## RAG Architecture 

Documents: BDDK AML rehberleri, FATF 40 Recommendations, MASAK SAR rehberi, EU AI Act
Chunking: RecursiveCharacterTextSplitter, 512 tokens, 128 overlap
Embedding: text-embedding-3-large or BAAI/bge-m3 (multilingual)
Vector store: Qdrant (self-hosted, localhost:6333)
Retrieval: BM25 + dense hybrid → bge-reranker-v2-m3
Citation: source document + page number mandatory in every answer

## Tech Stack

Runtime:        Python 3.11, FastAPI, uvicorn, asyncpg
Validation:     Pydantic v2 strict mode
ORM:            SQLAlchemy 2.0 async + Alembic
Databases:      PostgreSQL 16, Redis 7, Qdrant
ML:             XGBoost 2.x, scikit-learn, SHAP
Agents:         LangGraph, LangChain, langchain-google-genai
LLMs:           claude-haiku-4.5 (investigation), claude-haiku-4.5 (critical/synthesis/SAR)
Observability:  LangSmith, MLflow, structlog
DevOps:         Docker Compose, GitHub Actions, uv, ruff, mypy
Demo:           Streamlit (Hafta 7)
Deployment:     Hugging Face Spaces (Hafta 8)

## Current State

Hafta 1 ✓  Project skeleton, Docker stack, GitHub Actions CI
Hafta 2 ✓  XGBoost baseline + Optuna tuning (PR-AUC 0.4834), SHAP analysis
Hafta 3 ✓  FastAPI backend, Pydantic schemas, SQLAlchemy/Alembic, triage router, 18/18 integration tests
Hafta 4 ✓  Investigation Agent (claude-haiku-4.5), 5 tools, LangGraph graph, LangSmith tracing
Hafta 5 ✓  RAG pipeline, Qdrant indexing, Critical Agent (claude-haiku-4.5)
Hafta 6 ✓  Decision Synthesizer, SAR Generator
Hafta 7    Eval framework, Streamlit demo, Grafana
Hafta 8    Deployment (HF Spaces), README polish, blog post

## Key Design Decisions

1. Triage Router is rule-based — routing is a business rule, not an AI decision
2. XGBoost for tabular scoring — deterministic, auditable, 100x faster and cheaper than LLM
3. SHAP on every prediction — explainability is not optional in AML context
4. claude-haiku-4.5 for investigation (~30% volume), gemini-2.5-pro for critical (~10%) — cost-conscious routing
5. Feature extractor maps banking API fields to IEEE-CIS features via pre-computed JSON rule files
6. raw_mode=true on POST /transactions accepts IEEE-CIS features directly — used for testing and demo
7. Pydantic strict + retry on all LLM outputs — hallucination defense
8. Mock tools return deterministic data (hashlib seed) — reproducible tests without external deps
9. SHAP values injected at agent-creation time (factory pattern) — no DB round-trip during ReAct loop
10. InvestigationResult has INCONCLUSIVE fallback — agent never crashes the pipeline