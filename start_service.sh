#!/usr/bin/env sh
set -eu

ROLE="${SERVICE_ROLE:-backend}"

if [ "$ROLE" = "frontend" ]; then
  exec streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port="${PORT:-8501}"
fi

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
