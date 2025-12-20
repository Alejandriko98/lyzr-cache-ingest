from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from lyzr.client import LyzrClient
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
        # Inicializar cliente Lyzr
        client = LyzrClient(api_key=os.environ.get("LYZR_API_KEY"))
        
        # Crear contenido del chunk
        chunk_content = f"Pregunta: {request.query}\n\nRespuesta: {request.answer}"
        
        # Definir metadatos
        metadata = {
            "fecha_ingesta": datetime.now().isoformat(),
            "fuente_url": request.source_url,
            "tipo_contenido": "cache_web_search",
            "query_original": request.query
        }
        
        # Ingestar en la Base de Conocimiento
        result = client.knowledge_base.ingest_text(
            kb_id="68c7c30ec15ac5dc88995c1c",
            text=chunk_content,
            metadata=metadata
        )
        
        return {
            "status": "success",
            "message": "Resultado ingestado exitosamente en el caché semántico",
            "chunk_id": result.get("id", "N/A"),
            "timestamp": metadata["fecha_ingesta"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ingestar: {str(e)}")

@app.get("/")
async def root():
    return {"message": "API de Caché Semántico activa"}
