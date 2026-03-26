# FloatChat AI - Deployment Guide

## Railway (Recommended)

Before your first deploy, ensure build context is small:
- Keep `.dockerignore` in repository root.
- Exclude local folders such as `venv/`, `.venv/`, `chroma_db/`, `data/`, and `logs/`.
- Deploy from GitHub (not local upload) so ignored files are not sent to build context.

### 1. Backend Service

1. Create a new Railway service from this repo.
2. Attach PostgreSQL plugin.
3. Railway uses `railway.json` startup script. Set `SERVICE_ROLE=backend`.
4. Set environment variables:

```env
DATABASE_URL=<Railway Postgres URL>
LLM_PROVIDER=openrouter
LLM_MODEL=qwen/qwen3-8b:free
OPENROUTER_API_KEY=<your key>
VECTOR_STORE=memory
MAX_DOCUMENTS=24000
```

5. Verify health endpoint:

```bash
curl https://<backend>.up.railway.app/health
```

### 2. Frontend Service (Streamlit)

Create a second Railway service from the same repository and set env variable:

```env
SERVICE_ROLE=frontend
```

Set frontend env:

```env
BACKEND_URL=https://<backend>.up.railway.app
```

## Free Online LLM Options

- `openrouter` (recommended): supports free models including Qwen variants.
- `groq`: free tier with fast inference.
- `openai`: paid fallback.
- `ollama`: local development fallback.

## Dataset Ingestion (Seanoe / GDAC)

Use real global Argo data source referenced by:
- Seanoe DOI page: `https://www.seanoe.org/data/00311/42182/`

Run ingestion:

```bash
python pipeline/ingest_seanoe_argo.py
python pipeline/data_chroma_floats.py
```

Tune with env variables:

```env
ARGO_MAX_PROFILES=250
ARGO_GDAC_HTTP_BASE=https://data-argo.ifremer.fr/
ARGO_INDEX_PATH=argo_synthetic-profile_index.txt
```

## Troubleshooting

- `LLM request failed`: check API key for the selected `LLM_PROVIDER`.
- Slow startup on cloud: first start downloads embeddings model.
- Chroma persistence on Railway: prefer `VECTOR_STORE=memory`.
- `Image size exceeded limit`: verify `.dockerignore` is present and that heavy local folders are excluded from build context.
