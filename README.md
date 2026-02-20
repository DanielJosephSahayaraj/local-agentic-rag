# Local RAG Agent Pipeline

End-to-end Retrieval-Augmented Generation system built from scratch.

## Features
- PDF loading & semantic chunking
- FAISS vector store + hybrid retrieval (dense + BM25)
- Reranking with FlashRank
- Query rewriting + HyDE
- Semantic caching (hit rate improvement)
- Basic router (retrieve / no-retrieve)
- Offline evaluation with RAGAS
- Agentic retry loop (self-check & re-retrieve)

## Tech stack
- Embeddings: all-MiniLM-L6-v2
- LLM: Llama-3.1-8B-Instruct (via Ollama)
- Vector DB: FAISS
- Evaluation: RAGAS

## Setup
1. `pip install -r requirements.txt`
2. `ollama pull llama3.1:8b-instruct-q5_K_M`
3. Run `python main.py` or `streamlit run app.py`

## Results
- Hybrid retrieval improved faithfulness by X% (see RAGAS comparison)
- Cache hit rate: ~XX% on repeated/similar queries

Work in progress — adding Streamlit UI and more guardrails.