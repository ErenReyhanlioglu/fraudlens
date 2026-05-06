# Katkıda Bulunma

## Geliştirme Ortamı Kurulumu

```bash
# Bağımlılıkları kur (uv gerekli)
uv sync --all-groups

# Servisleri başlat
docker compose up -d postgres redis qdrant

# .env dosyasını oluştur
cp .env.example .env
# ANTHROPIC_API_KEY değerini ekle
```

## Kod Kuralları

Bu proje `ruff` kullanır. Commit öncesi:

```bash
uv run ruff check . --fix
uv run ruff format .
```

**Kurallar:**
- Maksimum satır uzunluğu: 100 karakter
- Tür ipuçları zorunlu (Pydantic v2 strict)
- `print()` yasak — `structlog` kullan
- FastAPI ve SQLAlchemy için `async/await` zorunlu
- Yorum yalnızca neden açık değilse yazılır; ne yaptığını açıklayan yorum yazılmaz

## Testler

```bash
# Sağlık kontrolleri (API çalışıyor olmalı)
uv run python scripts/critical_agent_healthcheck.py
uv run python scripts/investigator_agent_healthcheck.py
uv run python scripts/sar_agent_healthcheck.py

# Değerlendirme paketi
uv run tests/eval/run_eval.py
```

## PR Süreci

1. `main` dalından yeni bir dal aç: `git checkout -b feat/özellik-adı`
2. Değişiklikleri yap, `ruff` ile kontrol et
3. PR aç — başlık kısa (70 karakter), açıklama detaylı
4. CI geçmeden merge yapılmaz
