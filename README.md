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

<h1 align="center">FraudLens</h1>

<p align="center">
  Çok ajanlı AML dolandırıcılık tespiti · XGBoost + LangGraph + RAG
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-ReAct-6C63FF?style=flat" />
  <img src="https://img.shields.io/badge/XGBoost-2.x-FF6600?style=flat" />
  <img src="https://img.shields.io/badge/Claude-Haiku-D97706?style=flat" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/HF%20Spaces-Demo-FFD21E?style=flat&logo=huggingface&logoColor=black" />
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/ErenReyhanlioglu/fraudlens">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Canl%C4%B1%20Demo-HF%20Spaces-blue?style=for-the-badge" />
  </a>
</p>

---

## Nedir?

FraudLens, banka işlemlerini gerçek zamanlı analiz eden 6 katmanlı çok ajanlı bir AML (Kara Para Aklama Önleme) sistemidir. XGBoost modeli her işlemi puanlar; şüpheli işlemler BDDK/FATF mevzuatına dayanan LangGraph ajanlarına yönlendirilir. Yüksek riskli durumlar için otomatik SAR (Şüpheli İşlem Raporu) üretilir.

Sistem üç temel ilkeye dayanır: **hız** (ML katmanı belirleyici ve hızlı), **açıklanabilirlik** (her karar SHAP değerleriyle ve mevzuat atıflarıyla desteklenir), **güvenilirlik** (Pydantic strict mod + retry + INCONCLUSIVE fallback).

---

## Mimari

```
POST /api/v1/transactions
        │
        ▼
  ┌─────────────┐
  │  XGBoost    │  <50ms · 79 özellik · TreeSHAP
  └──────┬──────┘
         │
    ┌────▼────┐
    │ Triage  │  p < 0.30 → onayla
    └────┬────┘  0.30–0.70 → Investigation Agent
         │       p ≥ 0.70  → Critical Agent
    ┌────┴─────────────────────┐
    │                          │
┌───▼──────────┐   ┌───────────▼──────────┐
│ Investigation│   │  Critical Agent       │
│ Agent        │   │  + RAG (BDDK/FATF)   │
│ (Haiku)      │   │  + Network Analysis   │
└───────┬──────┘   └───────────┬──────────┘
        └──────────┬───────────┘
                   ▼
          ┌────────────────┐
          │  Synthesizer   │  FraudDecision üretir
          └───────┬────────┘
                  │
          ┌───────▼────────┐
          │ SAR Generator  │  Yalnızca escalate kararlarında
          └────────────────┘
```

| Katman | Bileşen | Model / Teknoloji | Kapsam |
|--------|---------|-------------------|--------|
| 0 | FastAPI endpoint | Python 3.11, asyncpg | Tüm istekler |
| 1 | Anomali Puanlayıcı | XGBoost, TreeSHAP | Her işlem |
| 2 | Triage Router | Kural tabanlı | Her işlem |
| 3a | Investigation Agent | Claude Haiku + LangGraph | ~%30 |
| 3b | Critical Agent | Claude Haiku + RAG | ~%10 |
| 4 | Decision Synthesizer | Claude Haiku | Tüm ajan kararları |
| 5 | SAR Generator | Claude Haiku | Yalnızca escalate |
| 6 | Gözlemlenebilirlik | LangSmith, MLflow, structlog | Tam sistem |

---

## Teknoloji Yığını

| Kategori | Teknoloji |
|----------|-----------|
| Runtime | Python 3.11, FastAPI, uvicorn |
| Doğrulama | Pydantic v2 strict mode |
| ORM | SQLAlchemy 2.0 async + Alembic |
| Veritabanları | PostgreSQL 16, Redis 7, Qdrant |
| ML | XGBoost 2.x, scikit-learn, SHAP |
| Ajan Çerçevesi | LangGraph ReAct, LangChain |
| LLM | Claude Haiku 4.5 (tüm ajan rolleri) |
| RAG | Qdrant, bge-m3 embeddings, chunker |
| Gözlemlenebilirlik | LangSmith, MLflow, structlog |
| DevOps | Docker Compose, uv, ruff |
| Demo | Streamlit, Hugging Face Spaces |

---

## Hızlı Başlangıç

### A. Docker 

```bash
# 1. Yapılandırma
cp .env.example .env
# .env dosyasına ANTHROPIC_API_KEY ekle

# 2. Altyapıyı başlat (PostgreSQL, Redis, Qdrant, MLflow)
docker compose up -d

# 3. Veritabanı migration
uv run alembic upgrade head

# 4. API'yi başlat
uv run uvicorn src.fraudlens.api.main:app --reload --port 8001
```

### B. Manuel Kurulum

**Gereksinimler:** Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker (servisler için)

```bash
# Bağımlılıkları kur
uv sync

# Servisleri Docker ile başlat (sadece altyapı)
docker compose up -d postgres redis qdrant mlflow

# .env dosyasını oluştur
cp .env.example .env
# ANTHROPIC_API_KEY değerini ekle

# Migration ve API
uv run alembic upgrade head
uv run uvicorn src.fraudlens.api.main:app --reload --port 8001
```

### RAG İndeksini Oluştur

```bash
# BDDK/FATF PDF dokümanlarını data/docs/ altına koy
uv run python scripts/build_rag_index.py
```

---

## Proje Yapısı

```
src/fraudlens/
├── api/        # FastAPI route'ları, middleware
├── agents/     # LangGraph ajanları + araçlar
├── core/       # Pydantic Settings, logging, exception'lar
├── db/         # SQLAlchemy modelleri, Alembic migration'ları
├── ml/         # XGBoost, SHAP, feature_extractor, model serving
├── rag/        # Qdrant, chunker, embedder, retriever
└── schemas/    # Pydantic modelleri: transaction, decision, investigation, sar

demo/
├── app.py          # Streamlit demo (canlı API)
├── app_hf.py       # HF Spaces versiyonu (önceden hesaplanmış)
├── fixtures.py     # 5 sentetik senaryo (LOW/MEDIUM/HIGH)
└── assets/style.css

scripts/
├── build_rag_index.py
├── critical_agent_healthcheck.py
├── investigator_agent_healthcheck.py
└── sar_agent_healthcheck.py

tests/eval/         # Değerlendirme paketi
```

---

## API Referansı

| Metod | Endpoint | Açıklama |
|-------|----------|----------|
| `POST` | `/api/v1/transactions` | İşlem analizi (tam boru hattı) |
| `GET` | `/api/v1/transactions/{id}` | İşlem sonucunu getir |
| `GET` | `/health` | Sistem sağlık kontrolü |
| `GET` | `/docs` | Swagger UI |

**Örnek istek:**
```bash
curl -X POST http://localhost:8001/api/v1/transactions \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "txn_001", "amount": 15000, "currency": "TRY", ...}'
```

`raw_mode=true` parametresiyle IEEE-CIS formatında özellikler doğrudan gönderilebilir (test/demo için).

---

## Çalışan Servisler

| Servis | URL | Açıklama |
|--------|-----|----------|
| FastAPI | http://localhost:8001 | Ana API |
| Swagger UI | http://localhost:8001/docs | Etkileşimli API dokümantasyonu |
| MLflow | http://localhost:5000 | Model takibi ve registry |
| Qdrant | http://localhost:6333/dashboard | Vektör veritabanı |
| PostgreSQL | localhost:5432 | İlişkisel veritabanı |
| Redis | localhost:6379 | Önbellek |

---

## Demo

**Canlı (API gerektirmez):** [HF Spaces](https://huggingface.co/spaces/ErenReyhanlioglu/fraudlens) — 5 önceden hesaplanmış senaryo (LOW approve, MEDIUM manual review, MEDIUM decline, HIGH escalate + SAR).

**Yerel (canlı API ile):**
```bash
uv run streamlit run demo/app.py
```

---

## Değerlendirme

```bash
# API çalışıyor olmalı
uv run tests/eval/run_eval.py --concurrency 3
# Sonuçlar: tests/eval/results/ altına kaydedilir
```

---

## Geliştirme

```bash
# Linting
uv run ruff check .
uv run ruff format --check .

# Otomatik düzeltme
uv run ruff check . --fix
uv run ruff format .

# Sağlık kontrolleri
uv run python scripts/critical_agent_healthcheck.py
uv run python scripts/investigator_agent_healthcheck.py
```

---

## Katkıda Bulunma

Detaylar için [CONTRIBUTING.md](CONTRIBUTING.md) dosyasına bakın.
