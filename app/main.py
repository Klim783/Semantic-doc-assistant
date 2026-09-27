import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from data.document_loader import DocumentLoader
from src.text_splitter import CharacterTextSplitter
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine

app = FastAPI(
	title = "Semantic document interlligence and RAG api",
	version = "1.0.0",
)

vector_store = VectorStore()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
rag_engine = RAGEngine(vector_store)

upload_dir = "data/uploads"
os.makedirs(upload_dir, exist_ok = True)

class QueryRequest(BaseModel):
	query:str
	top_k:int = 3

@app.get("/")
async def root():
	return {"Status":"online", "indexed_chinks":vector_store.index.ntotal}

@app.post("/api/v1/upload")
async def upload_document(file:UploadFile = File(...)):
	file_path = os.path.join(upload_dir, file.filename)
	try:
		with open(file_path, "rb") as buffer:
			shutil.copyfileobj(file.file, buffer)
		text = DocumentLoader.load_document(file_path)
		chunks = splitter.split_text(text)
		vector_store.add_texts(chunks)
		vector_store.save()

		return {
			"message":f"Successfully processed '{file.filename}'",
			"extracted_characters":len(text),
			"created_chunks":len(chunks),
			"total_indexed_vectors":vector_store.index.ntotal
		}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/query")
async def query_document(payload:QueryRequest):
	if not payload.query.strip():
		raise HTTPException(status_code=400, detail="No query provided")
	result = rag_engine.query(user_query=payload.query, top_k = payload.top_k)
	return result