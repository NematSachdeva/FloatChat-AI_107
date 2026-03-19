"""
FloatChat AI — FastAPI Backend
RAG Pipeline: NetCDF → PostgreSQL → ChromaDB → LLM → Frontend
"""

import re
import os
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import config
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import create_engine

print(f"PORT env var: {os.environ.get('PORT', 'NOT SET')}")

# Imported lazily inside lifespan to avoid blocking port binding
export_to_ascii = export_to_netcdf = export_to_csv = None
NLToSQLTranslator = process_analytical_query = None

# ── Globals (populated in lifespan) ──────────────────────────────────────────
engine     = None
collection = None
nl_translator = None


# ── Background init (runs AFTER port is open) ─────────────────────────────────
async def _background_init():
    """Load all heavy components after the port is already bound."""
    global engine, collection, nl_translator
    global export_to_ascii, export_to_netcdf, export_to_csv
    global NLToSQLTranslator, process_analytical_query

    try:
        from export_utils import (
            export_to_ascii as _ea,
            export_to_netcdf as _en,
            export_to_csv as _ec,
        )
        export_to_ascii, export_to_netcdf, export_to_csv = _ea, _en, _ec
        print("Export utils loaded")
    except Exception as e:
        print(f"Export utils import failed: {e}")

    try:
        from nl_to_sql import NLToSQLTranslator as _NL, process_analytical_query as _paq
        NLToSQLTranslator, process_analytical_query = _NL, _paq
        print("NL-to-SQL module loaded")
    except Exception as e:
        print(f"NL-to-SQL import failed: {e}")

    try:
        engine = create_engine(config.DATABASE_URL)
        print("Database engine created")
    except Exception as e:
        print(f"Database init failed: {e}")

    try:
        if NLToSQLTranslator:
            nl_translator = await asyncio.to_thread(NLToSQLTranslator)
            print("NL→SQL translator ready")
    except Exception as e:
        print(f"NL translator init failed: {e}")

    # ChromaDB + embeddings (downloads ~90 MB model — done post-startup)
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        _embed_model = await asyncio.to_thread(
            SentenceTransformer, "sentence-transformers/all-MiniLM-L6-v2"
        )

        class _EmbedFn:
            def __call__(self, input):
                if isinstance(input, str):
                    input = [input]
                return _embed_model.encode(input).tolist()

        _ef = _EmbedFn()

        if config.VECTOR_STORE == "persistent":
            os.makedirs(config.CHROMA_PATH, exist_ok=True)
            _chroma = chromadb.PersistentClient(path=config.CHROMA_PATH)
        else:
            _chroma = chromadb.EphemeralClient()

        collection = _chroma.get_or_create_collection(
            name="argo_measurements", embedding_function=_ef
        )
        print(f"ChromaDB ready — {collection.count():,} documents")
    except Exception as e:
        print(f"ChromaDB init failed: {e}")
        collection = None

    print("Background init complete.")


# ── Lifespan: yields immediately so port opens, then fires background init ────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FloatChat AI starting — port binding now.")
    asyncio.create_task(_background_init())
    yield
    print("Shutting down.")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FloatChat AI",
    description="RAG pipeline for ARGO oceanographic float data",
    lifespan=lifespan,
)


# ── Query classifier ──────────────────────────────────────────────────────────
_CHAT_RE = re.compile(
    r"^(hi+|hello+|hey+|howdy|sup|what'?s up|good\s+(morning|afternoon|evening|day)|"
    r"how are you|who are you|what are you|what can you do|"
    r"thanks?|thank you|okay|ok|cool|nice|great|wow|lol|haha|"
    r"bye|goodbye|see you|yes|no|sure|alright|got it|sounds good)[!?.]*$",
    re.IGNORECASE,
)

_DATA_RE = re.compile(
    r"\b(temperature|salinity|depth|pressure|oxygen|chlorophyll|ph|"
    r"argo|float|profile|cycle|deployment|wmo|measurement|observation|sensor|"
    r"ocean|sea|indian|pacific|atlantic|arabian|bengal|southern|"
    r"region|lat|lon|latitude|longitude|surface|deep|bgc|"
    r"data|show|find|analyze|compare|average|mean|trend|distribution)\b",
    re.IGNORECASE,
)


def classify(query: str) -> str:
    q = query.strip()
    if _CHAT_RE.match(q):
        return "chat"
    if _DATA_RE.search(q):
        return "data"
    if len(q.split()) <= 3:
        return "chat"
    return "data"


# ── LLM call (non-blocking) ───────────────────────────────────────────────────
def _llm_sync(messages: list) -> str:
    provider = config.LLM_PROVIDER.lower()

    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            max_tokens=1024,
        )
        return resp.choices[0].message.content

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            max_tokens=1024,
        )
        return resp.choices[0].message.content

    # Default: Ollama (local)
    import ollama as _ollama
    resp = _ollama.chat(model=config.LLM_MODEL, messages=messages)
    return resp["message"]["content"]


async def llm(messages: list) -> str:
    return await asyncio.to_thread(_llm_sync, messages)


# ── RAG retrieval + answer ────────────────────────────────────────────────────
async def rag_answer(query: str) -> dict:
    if collection is None:
        return {
            "answer": "Vector database is unavailable. Please check ChromaDB.",
            "context_documents": [],
            "retrieved_metadata": [],
        }

    results = collection.query(query_texts=[query], n_results=20)
    docs  = results["documents"][0]
    metas = results["metadatas"][0]

    rows = []
    for i, (_, m) in enumerate(zip(docs, metas), 1):
        row = f"[{i}]"
        if m.get("float_id"):                row += f" Float={m['float_id']}"
        if m.get("date"):                    row += f" | Date={m['date']}"
        if m.get("latitude")  is not None:   row += f" | Lat={m['latitude']:.2f}"
        if m.get("longitude") is not None:   row += f" | Lon={m['longitude']:.2f}"
        if m.get("depth")     is not None:   row += f" | Depth={m['depth']:.0f}m"
        if m.get("temperature") is not None: row += f" | Temp={m['temperature']:.2f}°C"
        if m.get("salinity")    is not None: row += f" | Sal={m['salinity']:.2f} PSU"
        if m.get("oxygen")      is not None: row += f" | O2={m['oxygen']:.2f} ml/L"
        rows.append(row)

    context = "\n".join(rows)

    temps = [m["temperature"] for m in metas if m.get("temperature") is not None]
    sals  = [m["salinity"]    for m in metas if m.get("salinity")    is not None]
    stats = []
    if temps:
        stats.append(
            f"Temperature — avg: {sum(temps)/len(temps):.2f}°C, "
            f"min: {min(temps):.2f}°C, max: {max(temps):.2f}°C ({len(temps)} readings)"
        )
    if sals:
        stats.append(
            f"Salinity — avg: {sum(sals)/len(sals):.2f} PSU, "
            f"min: {min(sals):.2f} PSU, max: {max(sals):.2f} PSU ({len(sals)} readings)"
        )
    stats_block = "\n".join(stats) if stats else "No numeric data in retrieved results."

    system = (
        "You are FloatChat AI, an expert oceanographic data assistant. "
        "You have access to real ARGO float measurements retrieved from a scientific database.\n\n"
        "STRICT RULES:\n"
        "1. Answer ONLY using the data provided below. Do not invent or assume values.\n"
        "2. If the retrieved data does not contain enough information, clearly say so.\n"
        "3. Always cite actual numbers from the data (temperature, salinity, depth, location).\n"
        "4. Be concise, factual, and scientifically accurate.\n"
        "5. You may add brief oceanographic context but mark it as general knowledge, not from the data."
    )

    user_msg = (
        f"Retrieved ARGO measurements ({len(docs)} observations):\n"
        f"{context}\n\n"
        f"Statistics from retrieved data:\n"
        f"{stats_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the data above:"
    )

    answer = await llm([
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ])

    return {
        "answer": answer,
        "context_documents": docs,
        "retrieved_metadata": metas,
    }


# ── SQL analytical answer ─────────────────────────────────────────────────────
async def sql_answer(query: str) -> Optional[dict]:
    if nl_translator is None:
        return None
    try:
        if not nl_translator.is_analytical_query(query):
            return None
        sql_result, error = process_analytical_query(query)
        if error or sql_result is None:
            return None
        df = sql_result.get("results")
        if df is None or df.empty:
            return None

        intent  = sql_result.get("intent", "unknown")
        sql_str = sql_result.get("sql_query", "")
        preview = df.head(20).to_string(index=False)

        system = (
            "You are FloatChat AI, an oceanographic data analyst. "
            "Summarize the SQL query results clearly and concisely. "
            "Use the actual numbers. Do not add data that is not in the table."
        )
        user_msg = (
            f"Query: {query}\n\n"
            f"SQL results ({len(df)} rows):\n{preview}\n\n"
            f"Write a clear, concise summary with the key numbers and findings:"
        )

        answer = await llm([
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ])

        return {
            "answer": answer,
            "context_documents": [f"SQL ({intent}): {sql_str[:300]}"],
            "retrieved_metadata": [{
                "query_type": "analytical",
                "intent": intent,
                "row_count": len(df),
            }],
            "sql_results": df.head(100).to_dict("records"),
        }
    except Exception as e:
        print(f"SQL path error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {"status": "ok", "service": "FloatChat AI"}


@app.get("/health")
async def health_check():
    chroma_ok = collection is not None
    return {
        "status":   "healthy",
        "database": "connected" if engine else "disconnected",
        "chromadb": "connected" if chroma_ok else "disconnected",
        "llm":      config.LLM_MODEL,
        "docs":     collection.count() if chroma_ok else 0,
    }


class QueryRequest(BaseModel):
    query_text: str


class QueryResponse(BaseModel):
    answer: str
    context_documents: list
    retrieved_metadata: list
    sql_results: Optional[list] = None


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    query = request.query_text.strip()
    if not query:
        return QueryResponse(
            answer="Please enter a question.",
            context_documents=[],
            retrieved_metadata=[],
        )

    kind = classify(query)

    if kind == "chat":
        system = (
            "You are FloatChat AI, a friendly assistant for an ARGO oceanographic "
            "data platform. Answer naturally and helpfully. Keep responses short."
        )
        answer = await llm([
            {"role": "system", "content": system},
            {"role": "user",   "content": query},
        ])
        return QueryResponse(
            answer=answer,
            context_documents=[],
            retrieved_metadata=[{"query_type": "chat"}],
        )

    sql_result = await sql_answer(query)
    if sql_result:
        return QueryResponse(**sql_result)

    rag_result = await rag_answer(query)
    return QueryResponse(**rag_result)


class ProfileRequest(BaseModel):
    ids: list[int]


@app.post("/get_profiles")
async def get_profiles_by_ids(request: ProfileRequest):
    if not request.ids:
        return []
    try:
        ids_tuple = tuple(request.ids)
        sql = """
            SELECT m.id, m.time, m.lat, m.lon, m.depth,
                   m.temperature, m.salinity, m.oxygen, m.ph, m.chlorophyll,
                   m.float_id, m.profile_id, p.cycle_number, f.wmo_id
            FROM measurements m
            JOIN profiles p ON m.profile_id = p.profile_id
            JOIN floats   f ON m.float_id   = f.float_id
            WHERE m.id IN %s
            ORDER BY m.float_id, p.cycle_number, m.depth;
        """
        df = pd.read_sql_query(sql, engine, params=(ids_tuple,))
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


@app.get("/float/{float_id}")
async def get_float_info(float_id: str):
    try:
        float_df = pd.read_sql_query(
            "SELECT * FROM floats WHERE float_id = %s;", engine, params=(float_id,)
        )
        if float_df.empty:
            return {"error": "Float not found"}
        profile_summary = pd.read_sql_query(
            "SELECT COUNT(*) as total_profiles, MIN(profile_date) as first_profile, "
            "MAX(profile_date) as last_profile FROM profiles WHERE float_id = %s;",
            engine, params=(float_id,)
        )
        measurement_summary = pd.read_sql_query(
            "SELECT COUNT(*) as total_measurements, MIN(depth) as min_depth, "
            "MAX(depth) as max_depth, AVG(temperature) as avg_temp, "
            "AVG(salinity) as avg_sal FROM measurements WHERE float_id = %s;",
            engine, params=(float_id,)
        )
        return {
            "float_info":           float_df.to_dict("records")[0],
            "profile_summary":      profile_summary.to_dict("records")[0],
            "measurement_summary":  measurement_summary.to_dict("records")[0],
        }
    except Exception as e:
        return {"error": str(e)}


class ExportRequest(BaseModel):
    format: str
    data_ids: list[int]


@app.post("/export")
async def export_data(request: ExportRequest):
    try:
        if export_to_ascii is None:
            return {"error": "Export service unavailable"}
        fmt = request.format.lower()
        if fmt == "ascii":
            content   = export_to_ascii(request.data_ids)
            mime, ext = "text/plain", "txt"
        elif fmt == "netcdf":
            content   = export_to_netcdf(request.data_ids)
            mime, ext = "application/octet-stream", "nc"
        elif fmt == "csv":
            content   = export_to_csv(request.data_ids)
            mime, ext = "text/csv", "csv"
        else:
            return {"error": "Unsupported format. Use ascii, netcdf, or csv."}
        return Response(
            content=content, media_type=mime,
            headers={"Content-Disposition": f"attachment; filename=argo_data.{ext}"},
        )
    except Exception as e:
        return {"error": str(e)}


@app.get("/statistics/system")
async def system_statistics():
    try:
        stats = pd.read_sql_query(
            """SELECT
                 (SELECT COUNT(DISTINCT float_id) FROM floats)       AS active_floats,
                 (SELECT COUNT(*)                 FROM profiles)      AS total_profiles,
                 (SELECT COUNT(*)                 FROM measurements)  AS total_measurements,
                 (SELECT ROUND(AVG(temperature)::numeric,2) FROM measurements
                  WHERE temperature IS NOT NULL)                      AS avg_temperature,
                 (SELECT ROUND(AVG(salinity)::numeric,2) FROM measurements
                  WHERE salinity IS NOT NULL)                         AS avg_salinity
            """,
            engine,
        )
        row = stats.to_dict("records")[0]
        row["data_quality"] = 95.0
        return row
    except Exception as e:
        return {"error": str(e)}
