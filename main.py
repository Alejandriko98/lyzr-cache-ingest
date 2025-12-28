from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
from io import BytesIO

app = FastAPI()

# Configuración de Lyzr
LYZR_API_KEY = "sk-default-b1r05tfEMdzpWKARUkRAuRuPzV2Q8iVa"
LYZR_RAG_ID = "68c7c30ec15ac5dc88995c1c"
LYZR_ENDPOINT = "https://rag-prod.studio.lyzr.ai/v3/train/txt/"

class CacheData(BaseModel):
    query: str
    answer: str
    source_url: str

@app.get("/")
def read_root():
    return {"message": "Semantic Cache API running"}

@app.post("/ingest_cache")
async def ingest_cache(data: CacheData):
    try:
        # Crear contenido del archivo TXT
        txt_content = f"""Query: {data.query}
Answer: {data.answer}
Source: {data.source_url}
---"""
        
        # Preparar el archivo en memoria
        file_bytes = txt_content.encode('utf-8')
        
        # Enviar a Lyzr
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {'file': ('cache_entry.txt', BytesIO(file_bytes), 'text/plain')}
            headers = {'x-api-key': LYZR_API_KEY}
            params = {'rag_id': LYZR_RAG_ID}
            
            response = await client.post(
                LYZR_ENDPOINT,
                files=files,
                headers=headers,
                params=params,
                data={'data_parser': 'txt_parser', 'extra_info': '{}'}
            )
            
            response.raise_for_status()
            
        return {
            "status": "success",
            "message": "Cache ingested to Lyzr KB",
            "lyzr_response": response.json()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
