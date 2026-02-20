# Local RAG Agent Pipeline

End-to-end **Retrieval-Augmented Generation (RAG)** system built from scratch, running fully locally.

## Features
- PDF ingestion & intelligent chunking
- FAISS vector database + **hybrid retrieval** (dense semantic + BM25 keyword)
- Reranking with FlashRank/CrossEncoder
- Query rewriting + HyDE for short/vague questions
- Semantic caching (significant LLM call savings)
- Basic router (retrieve / no-retrieve decision)
- Offline evaluation with **RAGAS** (faithfulness, relevancy, precision, recall)
- Simple agentic retry loop (self-check & re-retrieve)

## Tech Stack
- Embeddings: `all-MiniLM-L6-v2`
- LLM: Llama-3.1-8B-Instruct (via Ollama, Q5_K_M quantization)
- Vector DB: FAISS
- Reranking: FlashRank / CrossEncoder
- Evaluation: RAGAS
- UI: Streamlit (in progress)

## Quick Start
1. Clone repo
   ```bash
   git clone https://github.com/DanielJosephSahayaraj/Local-Agentic-RAG-System_Evaluation-Driven_Built-with-FAISS-Ollama-RAGAS
   cd YOUR_REPO

## Demo
<image-card alt="RAG Chat UI" src="screenshot.png" ></image-card>
<image-card alt="RAG Terminal" src="screenshot1.png" ></image-card>

## Pipeline Flowchart

<image-card alt="RAG Agent Pipeline Flowchart" src="Agentic_RAG(Local).jpg" ></image-card>

High-level overview of the query → retrieval → generation → retry loop.