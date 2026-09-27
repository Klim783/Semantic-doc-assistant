from typing import List

class CharacterTextSplitter:
	def __init__(self, chunk_size:int = 500, chunk_overlap:int = 50):
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap

	def split_text(self, text:str) -> List[str]:
		if not text:
			return []
		chunks = []
		start = 0
		text_length = len(text)

		while start < text_length:
			end = start + self.chunk_size
			chunk = text[start:end]
			chunks.append(chunk.strip())
			start += self.chunk_size - self.chunk_overlap
		return chunks