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
    cache_key = make_cache_key(q.query, q.mode)

    # 1️⃣ INTENTAR CACHE
    cached_answer = redis_client.get(cache_key)
    if cached_answer:
        return {
            "mode": q.mode,
            "cached": True,
            "answer": cached_answer
        }

    # 2️⃣ SELECCIÓN DE MODELO Y PROMPT
    if q.mode == "pro":
        model = "gpt-4o"
        ttl = 60 * 60 * 24 * 7  # 7 días
        system_prompt = """
Eres AITAX Pro, consultor fiscal experto en España para autónomos, PYMES y sociedades.

Tu objetivo es ofrecer asesoramiento fiscal profesional, estratégico y bien estructurado, sin inventar información.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALCANCE Y RESPONSABILIDAD:
- Respondes con base en normativa fiscal española conocida y práctica habitual.
- Si la información puede haber cambiado, adviértelo claramente.
- Si no tienes certeza suficiente, indica que debe verificarse.

NUNCA inventes artículos, porcentajes ni beneficios fiscales.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ESTRUCTURA OBLIGATORIA:

📌 RESPUESTA RÁPIDA:
1–2 frases claras.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 ANÁLISIS DETALLADO:
- Puntos clave
- Riesgos
- Ejemplos cuando sea relevante

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 RECOMENDACIÓN ESTRATÉGICA:
Consejo profesional claro.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 REFERENCIAS LEGALES (si procede):
Solo si estás razonablemente seguro.
"""
    else:
        model = "gpt-4o-mini"
        ttl = 60 * 60 * 24  # 24 horas
        system_prompt = """
Eres AITAX, un asistente fiscal experto en España para autónomos y pequeños negocios.

Tu objetivo es ofrecer respuestas claras, prácticas y orientativas sobre fiscalidad básica.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FORMA DE RESPONDER:
- Lenguaje sencillo
- Explicaciones prácticas
- Sin tecnicismos innecesarios

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REGLAS:
- No inventes normativa ni cifras exactas
- Si no tienes certeza suficiente, dilo claramente
"""

    # 3️⃣ LLAMADA A OPENAI
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

    # 4️⃣ GUARDAR EN REDIS
    redis_client.setex(cache_key, ttl, answer)

    # 5️⃣ LOGS
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
