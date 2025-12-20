from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from lyzr_automata import Agent
from datetime import datetime
import os

app = FastAPI(title="Caché Semántico API")

class IngestRequest(BaseModel):
    query: str
    answer: str
    source_url: str

@app.post("/ingest_cache")
async def ingest_cache(request: IngestRequest):
    try:
        # Crear contenido del chunk
        chunk_content = f"Pregunta: {request.query}\n\nRespuesta: {request.answer}"
        
        # Código para ingestar usando lyzr-automata
        # (necesitas la documentación específica de este SDK)
        
        return {
            "status": "success",
            "message": "Resultado ingestado",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "API activa"}
