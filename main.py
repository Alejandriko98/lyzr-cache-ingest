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

    # 1️⃣ INTENTAR CACHE
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

    # 2️⃣ SELECCIÓN DE MODELO Y PROMPT
    if q.mode == "pro":
        model = "gpt-4o"
        ttl = 60 * 60 * 24 * 7  # 7 días
        system_prompt = """
Eres AITAX Pro, asesor fiscal senior en España para autónomos, PYMES y sociedades.

NO eres un chatbot generalista.
NO das respuestas académicas ni genéricas.
Actúas como un profesional contratado para analizar, decidir y orientar con criterio.

Tu objetivo es:
- bajar la fiscalidad a decisiones reales
- anticipar riesgos
- optimizar dentro de la legalidad
- aportar claridad cuando hay varias opciones

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRINCIPIOS DE RESPUESTA:
- Hablas con seguridad y criterio profesional.
- Evitas frases tipo “en general”, “se recomienda”, “conviene consultar”.
- NO derives al usuario a otros asesores: TÚ eres el asesor.
- Si algo depende de variables concretas, las explicas y acotas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CUÁNDO CITAR NORMATIVA:
- Cita leyes, artículos o consultas DGT SOLO si refuerzan el criterio.
- No abras la sección legal si no aportas valor real.
- No pongas “si procede”.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ESTRUCTURA OBLIGATORIA:

📌 RESPUESTA RÁPIDA  
Conclusión clara y accionable en 1–2 frases.
Debe responder a: “¿Qué haría yo en este caso?”

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 ANÁLISIS DETALLADO  
Aquí está el valor Pro:
- Qué variables importan de verdad
- Errores habituales
- Riesgos fiscales
- Escenarios posibles y consecuencias

Usa ejemplos solo si ayudan a decidir.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 CRITERIO PROFESIONAL  
Toma de posición clara:
- opción preferente
- por qué
- cuándo cambiarías de estrategia

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 REFERENCIAS LEGALES  
Inclúyelas SOLO si refuerzan el análisis.
Ejemplo: LIRPF, LIS, consultas DGT concretas.
Si no aportan, NO incluyas esta sección.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REGLAS CRÍTICAS:
- No uses disclaimers genéricos.
- No suavices conclusiones por miedo.
- No hables de limitaciones técnicas, planes o tokens.
- Mantén tono profesional, directo y seguro.
"""
    else:
        model = "gpt-4o-mini"
        ttl = 60 * 60 * 24 * 7  # 7 días
        system_prompt = """
Eres AITAX, asistente fiscal en España para autónomos y pequeños negocios.

Tu función es ayudar a entender obligaciones fiscales y decisiones habituales de forma clara y práctica.
NO eres un chatbot genérico.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FORMA DE RESPONDER:
- Lenguaje claro y directo
- Explicaciones prácticas
- Nada de frases vacías o académicas
- Responde como alguien que trabaja a diario con autónomos

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LÍMITES:
- Das orientación general, no planificación compleja
- Si algo depende de datos concretos, indícalo claramente
- No inventes cifras ni normativa exacta si no estás seguro

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ESTILO:
- Útil
- Claro
- Sin marketing
- Sin frases tipo “consulta con un asesor”
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

    # 5️⃣ MÉTRICA DE TOKENS
    redis_client.incrby("metrics:tokens_used", usage.total_tokens)

    # LOGS
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
