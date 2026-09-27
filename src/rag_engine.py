from typing import List, Tuple
from src.vector_store import VectorStore

class RAGEngine:
	def __init__(self, vector_store:VectorStore):
		self.vector_store = vector_store

	def build_prompt(self, query:str , context_chunk:List[str])->str:
		context_str = "\n---\n".join(context_chunk)
		prompt = f"""You are a helpful assistant answering user questions based strictly on the provided context documents.

CONTEXT INFORMATION:
{context_str}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question thoroughly using ONLY facts directly mentioned in the context above.
2. If the context does not contain enough information to answer the question, state: "The provided document does not contain this information."
3. Keep the answer clear, precise, and objective.

ANSWER:"""
		return prompt
	def query(self, user_query: str, top_k:int = 3) -> dict:
		retrieved_items:List[Tuple[str, float]] = self.vector_store.search(user_query, top_k = top_k)

		if not retrieved_items:
			return {
				"answer":"No documents indexed yet. Please upload a file first",
				"sources":[]
			}
		context_chunks = [item[0] for item in retrieved_items]
		prompt = self.build_prompt(user_query, context_chunks)
		return{
			"query":user_query,
			"prompt":prompt,
			"retrieved_context":context_chunks,
			"sources":[f"Chunk {i+1} (Distance: {item[1]:.4f})" for i, item in enumerate(retrieved_items)]
		}