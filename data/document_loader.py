import os
from pypdf import PdfReader

class DocumentLoader:
	@staticmethod
	def load_pdf(file_path: str) -> str:
		if not os.path.exists(file_path):
			raise FileNotFoundError(f"file not found at {file_path}")
		reader = PdfReader(file_path)
		extracted_text = []

		for page_num, page in enumerate(reader.pages):
			text = page.extract_text()
			if text:
				extracted_text.append(text)
		return "\n".join(extracted_text)

	@staticmethod
	def load_txt(file_path:str) -> str:
		with open(file_path, 'r', encoding = 'utf-8') as f:
			return f.read()

	@classmethod
	def load_document(cls, file_path:str) -> str:
		ext = os.path.splitext(file_path)[1].lower()
		if ext == ".pdf":
			return cls.load_pdf(file_path)
		elif ext == ".txt":
			return cls.load_txt(file_path)
		else:
			return ValueError(f"Not supported file type : {ext}")
		



