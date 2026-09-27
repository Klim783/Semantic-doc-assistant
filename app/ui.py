import streamlit as st
import requests

st.set_page_config(
	page_title="Semantic Doc Assistant",
	layout="wide"
)

API_URL = "http://127.0.0.1:8000"

st.title("Semantic Document Intelligence & RAG Assistant")
st.markdown("Upload your PDF or TXT documents and query them using natural language.")

# Sidebar for Document Management
st.sidebar.header("📁 Document Management")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
	if st.sidebar.button("Process Document"):
		with st.spinner("Extracting text, chunking, and generating embeddings..."):
			try:
				files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
				response = requests.post(f"{API_URL}/api/v1/upload", files=files)

				if response.status_code == 200:
					data = response.json()
					st.sidebar.success(f"File '{uploaded_file.name}' processed successfully!")
					st.sidebar.json(data)
				else:
					st.sidebar.error(f"Error: {response.json().get('detail')}")
			except Exception as e:
				st.sidebar.error(f"Failed to connect to backend service: {e}")

# Chat Session State Initialization
if "messages" not in st.session_state:
	st.session_state.messages = []

# Render Conversation History
for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])
		if "sources" in message and message["sources"]:
			with st.expander("Retrieved Context Chunks"):
				for src in message["sources"]:
					st.write(src)

# Handle New User Input
if prompt := st.chat_input("Ask a question about your documents..."):
	st.session_state.messages.append({"role": "user", "content": prompt})
	with st.chat_message("user"):
		st.markdown(prompt)

	with st.chat_message("assistant"):
		with st.spinner("Retrieving relevant context..."):
			try:
				res = requests.post(
					f"{API_URL}/api/v1/query",
					json={"query": prompt, "top_k": 3}
				)
				if res.status_code == 200:
					data = res.json()

					retrieved_chunks = data.get("retrieved_context", [])
					if retrieved_chunks:
						response_text = f"**Synthesized RAG Prompt for LLM:**\n\n```text\n{data.get('prompt')}\n```"
					else:
						response_text = "No vector data found. Please upload a document first."

					st.markdown(response_text)

					if retrieved_chunks:
						with st.expander("Retrieved Context Chunks"):
							for i, chunk in enumerate(retrieved_chunks, 1):
								st.write(f"**Chunk {i}:** {chunk}")

					st.session_state.messages.append({
						"role": "assistant",
						"content": response_text,
						"sources": retrieved_chunks
					})
				else:
					st.error("Failed to execute query on the server.")
			except Exception as e:
				st.error(f"Connection error: {e}")