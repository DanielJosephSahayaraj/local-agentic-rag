# Local Agentic RAG Pipeline

**100% offline Retrieval-Augmented Generation system** — built from scratch with Ollama, FAISS, and Streamlit.  
Chat with your PDFs/documents using a local LLM + hybrid retrieval + basic agentic retry.

<p align="center">
  <img src="assets/Agentic_RAG(Local).jpg" alt="Agentic RAG Pipeline Flowchart" width="80%">
  <br><em>High-level workflow: query understanding → hybrid retrieval → reranking → generation → self-check retry</em>
</p>

## ✨ Current Features (Early Stage)

- PDF/text ingestion + chunking
- Embeddings via `all-MiniLM-L6-v2` (or Ollama nomic-embed)
- **Hybrid search** — semantic (FAISS) + keyword (BM25 planned)
- Local LLM generation with **Llama-3.1-8B-Instruct** (via Ollama)
- Basic Streamlit chat UI
- In-progress: reranking, query rewrite/HyDE, semantic cache, RAGAS eval, agentic loop

## 🛠 Tech Stack

- **LLM** → Ollama + Llama 3.1 8B 
- **Embeddings** → sentence-transformers / Ollama
- **Vector DB** → FAISS
- **UI** → Streamlit
- **Others** → LangChain / LlamaIndex, rank_bm25, ragas 

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-green?style=for-the-badge&logo=ollama" alt="Ollama">
  <img src="https://img.shields.io/badge/FAISS-Vector%20DB-orange?style=for-the-badge" alt="FAISS">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Offline-100%25-success?style=for-the-badge" alt="Fully Offline">
</p>

