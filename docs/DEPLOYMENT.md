# FloatChat AI - Deployment Guide

## Quick Start

### Prerequisites
- Python 3.13+
- PostgreSQL
- Ollama
- Git

### Installation Steps

1. **Clone Repository**
```bash
git clone https://github.com/NematSachdeva/FloatChat-AI_107.git
cd FloatChat-AI_107/floatchat-ai
```

2. **Setup Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your settings
```

4. **Setup Database**
```bash
createdb argo
```

5. **Install Ollama Models**
```bash
ollama pull gemma2:2b
ollama pull nomic-embed-text:latest
```

6. **Start Services**

Terminal 1 - Ollama:
```bash
ollama serve
```

Terminal 2 - Backend:
```bash
./start_backend.sh
```

Terminal 3 - Frontend:
```bash
./start_frontend.sh
```

## Access Points

- Frontend: http://localhost:8501
- Backend API: http://127.0.0.1:8000
- API Docs: http://127.0.0.1:8000/docs

## Data Ingestion

After setup, ingest ARGO data:

```bash
python argo_float_processor.py
```

## Production Deployment

For production, use:
- Gunicorn for FastAPI
- Nginx as reverse proxy
- PostgreSQL with backups
- Docker for containerization

## Troubleshooting

- Backend 500 errors: Check Ollama is running
- ChromaDB errors: Reinitialize with `rm -rf chroma_db/`
- Database errors: Verify PostgreSQL connection

See README.md for detailed documentation.
