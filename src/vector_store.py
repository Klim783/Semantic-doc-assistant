import os
from typing import List, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
	"""Handles text embedding generation and FAISS vector index management using Cosine Similarity."""

	def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
		print(f"Loading Embedding Model: {model_name}...")
		self.encoder = SentenceTransformer(model_name)
		self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

		# 1. Используем IndexFlatIP (Inner Product) вместо IndexFlatL2
		self.index = faiss.IndexFlatIP(self.embedding_dim)
		self.chunks: List[str] = []

	def add_texts(self, texts: List[str]) -> None:
		if not texts:
			return

		print(f"Generating embeddings for {len(texts)} chunks...")
		embeddings = self.encoder.encode(
			texts,
			convert_to_numpy=True,
			normalize_embeddings=True,
			show_progress_bar=False
		)

		embeddings = np.array(embeddings).astype("float32")

		self.index.add(embeddings)
		self.chunks.extend(texts)
		print(f"Successfully indexed {self.index.ntotal} total vectors.")

	def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
		if self.index.ntotal == 0:
			return []

		# Нормализуем также вектор запроса
		query_vector = self.encoder.encode(
			[query],
			convert_to_numpy=True,
			normalize_embeddings=True
		).astype("float32")

		scores, indices = self.index.search(query_vector, top_k)

		results = []
		for idx, score in zip(indices[0], scores[0]):
			if idx != -1 and idx < len(self.chunks):
				results.append((self.chunks[idx], float(score)))

		return results

	def save(self, folder_path: str = "data") -> None:
		os.makedirs(folder_path, exist_ok=True)
		faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))

		import json
		with open(os.path.join(folder_path, "chunks.json"), "w", encoding="utf-8") as f:
			json.dump(self.chunks, f, ensure_ascii=False, indent=2)
		print(f"Vector store persisted to '{folder_path}/'.")

	def load(self, folder_path: str = "data") -> None:
		index_path = os.path.join(folder_path, "index.faiss")
		chunks_path = os.path.join(folder_path, "chunks.json")

		if os.path.exists(index_path) and os.path.exists(chunks_path):
			self.index = faiss.read_index(index_path)
			import json
			with open(chunks_path, "r", encoding="utf-8") as f:
				self.chunks = json.load(f)
			print(f"Loaded existing index with {self.index.ntotal} vectors.")