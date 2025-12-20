from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Caché Semántico API")

class IngestRequest(BaseModel):
    query: str
    answer: str
    source_url: str

@app.post("/ingest_cache")
async def ingest_cache(request: IngestRequest):
    try:
        chunk_content = f"Pregunta: {request.query}\n\nRespuesta: {request.answer}"
        
        # Aquí irá la integración con Lyzr cuando tengas las credenciales
        
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
