from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
import redis
import hashlib

# ---------- CONFIG ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")  # ej: redis://:password@host:port

client = OpenAI(api_key=OPENAI_API_KEY)

redis_client = redis.from_url(
    REDIS_URL,
    decode_responses=True  # strings en vez de bytes
)

app = FastAPI()

# ---------- MODELOS ----------
class Question(BaseModel):
    query: str
    mode: str = "standard"  # standard | pro

# ---------- UTILS ----------
def make_cache_key(query: str, mode: str) -> str:
    raw = f"{mode}:{query.lower().strip()}"
    return "aitax:" + hashlib.sha256(raw.encode()).hexdigest()

# ---------- HEALTH CHECK ----------
@app.get("/")
def root():
    return {"status": "AITAX agent running with Redis cache"}

# ---------- ENDPOINT PRINCIPAL ----------
@app.post("/ask")
def ask(q: Question):
    # Métrica total
    redis_client.incr("metrics:total_requests")

    cache_key = make_cache_key(q.query, q.mode)

    # 1️⃣ Intentar cache
    cached_answer = redis_client.get(cache_key)
    if cached_answer:
        redis_client.incr("metrics:cache_hits")
        return {
            "mode": q.mode,
            "cached": True,
            "answer": cached_answer
        }

    # Cache miss
    redis_client.incr("metrics:cache_misses")

    # 2️⃣ Selección de modelo y prompt
    if q.mode == "pro":
        model = "gpt-4o"
        ttl = 60 * 60 * 24 * 7  # 7 días
        system_prompt = """
Eres AITAX Pro, consultor fiscal senior en España para autónomos, PYMES y sociedades.

Actúas como un asesor humano experimentado: prudente, estratégico y orientado a minimizar riesgos fiscales.
Tu prioridad es la CORRECCIÓN y la UTILIDAD práctica, no impresionar.
"""
    else:
        model = "gpt-4o-mini"
        ttl = 60 * 60 * 24 * 7  # 7 días
        system_prompt = """
Eres AITAX, un asistente fiscal experto en España para autónomos y pequeños negocios.

Tu objetivo es ofrecer respuestas claras y orientativas sobre fiscalidad básica.
"""

    # 3️⃣ Llamada a OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": q.query}
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content
    usage = response.usage

    # 4️⃣ Guardar en Redis
    redis_client.setex(cache_key, ttl, answer)

    # 5️⃣ Métrica de tokens
    redis_client.incrby("metrics:tokens_used", usage.total_tokens)

    # Logs
    print("----- AITAX USAGE LOG -----")
    print("MODE:", q.mode)
    print("MODEL:", model)
    print("TOKENS:", usage.total_tokens)
    print("---------------------------")

    return {
        "mode": q.mode,
        "cached": False,
        "answer": answer,
        "tokens_used": usage.total_tokens
    }

@app.get("/metrics")
def metrics():
    return {
        "total_requests": int(redis_client.get("metrics:total_requests") or 0),
        "cache_hits": int(redis_client.get("metrics:cache_hits") or 0),
        "cache_misses": int(redis_client.get("metrics:cache_misses") or 0),
        "tokens_used": int(redis_client.get("metrics:tokens_used") or 0),
    }
