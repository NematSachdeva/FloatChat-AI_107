"""
Configuration for FloatChat
Supports both local development and cloud deployment
"""

import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database Configuration
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    if DB_PASSWORD:
        DATABASE_URL = f"postgresql+psycopg2://postgres:{quote_plus(DB_PASSWORD)}@localhost:5432/argo"
    else:
        DATABASE_URL = "postgresql+psycopg2://localhost:5432/argo"

# Render (and most cloud providers) give a plain postgres:// URL — SQLAlchemy needs the driver prefix
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
elif DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

# LLM Provider — "ollama" (local) | "groq" (cloud, free) | "openai" (cloud) | "openrouter" (cloud, free models)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Model name — depends on provider:
#   ollama  → gemma2:2b  (local)
#   groq    → llama-3.1-8b-instant  (free, fast)
#   openai  → gpt-4o-mini
#   openrouter → qwen/qwen3-8b:free
LLM_MODEL = os.getenv("LLM_MODEL", "gemma2:2b")

# API keys (only needed for cloud providers)
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "FloatChat AI")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Ollama Configuration (local only)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ChromaDB Configuration
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
VECTOR_STORE = os.getenv("VECTOR_STORE", "persistent")  # Options: persistent, memory

# Backend URL for frontend
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Data Processing Limits
MAX_FLOATS = int(os.getenv("MAX_FLOATS", "1000"))  # Increased for virtual floats from nc data
MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", "30000"))  # Limited to 30k as requested
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))

# Argo GDAC / Seanoe dataset ingestion
ARGO_DATA_SOURCE_URL = os.getenv("ARGO_DATA_SOURCE_URL", "https://www.seanoe.org/data/00311/42182/")
ARGO_GDAC_HTTP_BASE = os.getenv("ARGO_GDAC_HTTP_BASE", "https://data-argo.ifremer.fr/")
ARGO_INDEX_PATH = os.getenv("ARGO_INDEX_PATH", "argo_synthetic-profile_index.txt")
ARGO_MAX_PROFILES = int(os.getenv("ARGO_MAX_PROFILES", "250"))
ARGO_PROFILE_TIMEOUT_SECONDS = int(os.getenv("ARGO_PROFILE_TIMEOUT_SECONDS", "25"))

