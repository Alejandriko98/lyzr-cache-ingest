from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Question(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/ask")
def ask(q: Question):
    return {
        "answer": f"Pregunta recibida: {q.query}"
    }
