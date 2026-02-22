from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import Dict, Optional
from gpt4all import GPT4All
import pickle
from flashrank import Ranker, RerankRequest
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.language_models import LLM
from langchain_core.outputs import LLMResult
from datasets import Dataset
from rank_bm25 import BM25Okapi
import streamlit as  st
import os
import re
from typing import Optional
from datetime import datetime, timedelta

st.set_page_config(page_title="RAG Agent Chat", layout="wide")

CACHE_FILE = "rag_cache.pkl"
SIMILARITY_THRESHOLD = 0.95  
MAX_CACHE_SIZE = 1000
MAX_AGE_DAYS = 7
MAX_RETRIES = 2


@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = Ollama(
        model="llama3.1:8b", 
        temperature=0.7,
        num_predict=600
    )



    try:
        loader = PyPDFLoader("https://arxiv.org/pdf/2005.11401.pdf")
        documents = loader.load()
        print(f"Loaded {len(documents)} pages")
    except Exception as e:
        print("Failed to load PDF:", e)
        raise

    # ── Split ──
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # ↑ increased
        chunk_overlap=100     # ↑ increased
    )
    chunks = text_splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]
    print(f"Created {len(texts)} chunks")


    tokenized_texts = [text.lower().split() for text in texts]  # simple tokenization
    bm25 = BM25Okapi(tokenized_texts)

    # ── Embed ──
    embedding_vectors = embeddings.embed_documents(texts)


    embeddings_np = np.array(embedding_vectors).astype('float32')

    # ── FAISS index ──
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    print(f"Number of vectors in FAISS: {index.ntotal}")

    # ── Save ──
    faiss.write_index(index, "faiss_index.index")
    with open("texts.pkl", "wb") as f:
        pickle.dump(texts, f)

    print("Index saved successfully.")
    reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

    return embeddings, llm, index, texts, bm25, reranker

embeddings, llm, index, texts, bm25, reranker = load_resources()

# Load cache at startup (persistent from disk)
cache = {}
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded {len(cache)} cached items from disk")
    except Exception as e:
        print("Cache load failed (using empty cache):", e)

def normalize_query(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r'[^\w\s]', '', q)     
    q = ' '.join(q.split())            
    return q


def save_key(key: str, answer: str):
    q_vec = embeddings.embed_query(key)
    cache[tuple(q_vec)] = answer


def get_cached_response(query: str) -> Optional[str]:
    norm_q = normalize_query(query)
    q_vec = np.array(embeddings.embed_query(norm_q))

    now = datetime.utcnow()
    for cached_vec_tuple, value in list(cache.items()):
        cached_vec = np.array(cached_vec_tuple)
        sim = np.dot(q_vec, cached_vec) / (
            np.linalg.norm(q_vec) * np.linalg.norm(cached_vec) + 1e-10
        )
        if sim < SIMILARITY_THRESHOLD:
            continue
            
        if isinstance(value, str):
            print(f"Migrating old cache entry for sim={sim:.4f}")
            cache[cached_vec_tuple] = {"answer": value, "timestamp": now.isoformat()}
            return value
        
        if isinstance(value, dict):
            entry_time = datetime.fromisoformat(value["timestamp"])
            if now - entry_time <= timedelta(days=MAX_AGE_DAYS):
                print(f"Cache HIT! sim={sim:.4f}")
                return value["answer"]
            else:
                del cache[cached_vec_tuple]
                print("Expired cache entry removed")

    st.sidebar.info("Cache miss")
    return None




RETRY_STRATEGIES = [
    {"name": "Wider search", "k_vector": 20, "k_bm25": 20, "use_rewrite": True, "use_hyde": True},
    {"name": "Original query only", "k_vector": 12, "k_bm25": 12, "use_rewrite": False, "use_hyde": False},
    {"name": "No HyDE", "k_vector": 15, "k_bm25": 15, "use_rewrite": True, "use_hyde": False},
]

def save_to_cache(query: str, answer: str, rewrite_query: Optional[str] = None):
    if not answer or "don't know" in answer.lower() or len(answer.strip()) < 20:
        return
    
    timestamp = datetime.utcnow().isoformat()
    entry = {"answer": answer, "timestamp": timestamp}

    norm_orig = normalize_query(query)
    q_vec_orig = embeddings.embed_query(norm_orig)
    cache[tuple(q_vec_orig)] = entry

    if rewrite_query:
        norm_rew = normalize_query(rewrite_query)
        q_vec_rew = embeddings.embed_query(norm_rew)
        cache[tuple(q_vec_rew)] = entry


    if len(cache) > MAX_CACHE_SIZE:
        oldest_key = next(iter(cache))
        del cache[oldest_key]
        print("Cache full — removed oldest entry")


    # Save to disk
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
        print("→ Saved to persistent cache")
    except Exception as e:
        print("Cache save failed:", e)


def is_answer_good(question: str, answer: str, context: str) -> tuple[bool, str]:
    judge_prompt = f"""You are a very strict quality checker for RAG answers.

        Evaluate this answer for the given question and context:
        - Is it complete? (covers main points)
        - Is it faithful? (only uses info from context, no added facts)
        - Is it relevant, clear, and helpful?

        Question: {question}
        Answer: {answer}
        Context (truncated): {context[:1500]}...

        Reply with exactly this format:
        YES or NO
        One short sentence reason.

        Examples:
        YES - Answer is complete, faithful, and directly answers the question.
        NO - Answer adds information not present in the context.

        Your reply:"""

    judge_response = llm.invoke(judge_prompt).strip()

    # Simple parsing
    lines = judge_response.split("\n", 1)
    verdict = lines[0].strip().upper()
    reason = lines[1].strip() if len(lines) > 1 else "No reason given"

    is_good = "YES" in verdict
    print(f"Self-check verdict: {verdict} - {reason}")

    return is_good, reason




def run_rag_pipeline(query: str) -> str:
 
    cached = get_cached_response(query)

    if cached:
        final_response = cached
        print("\nFinal answer:")
        print(final_response)
    else:
        prompt_decision = f''' You are a binary classifier stictly do not use your knowledge to answer this. 

        Reply with exactly one word:
        RETRIEVE
        or
        NO_RETRIEVE

        wether the Question requires information from the database contains information about:
        - MLOps pipelines and best practices
        - Large language models (LLMs) and their architectures
        - Data processing, retrieval-augmented generation (RAG), and vector databases
        - Deployment, monitoring, and AI tool integrations


        strictly follow these 3 commands
        Do not explain.
        Do not add punctuation.
        Do not add spaces or new lines.


        Question:
        {query}


        Answer:

        '''
        response_decision = llm.invoke(
            prompt_decision)
        decision = response_decision.strip().upper()
        print(response_decision)

        if 'no_retrieve' in response_decision.lower():
            prompt = f''' You are a specialist assistant. You only have access to information 
            about MLOps and RAG. If the question is outside this scope, 
            politely inform the user you cannot answer.
                Question :
                {query}

            Answer :


            '''
            response = llm.invoke( prompt)
            max_tokens= 100
            print(response)

        else:

            rewrite_prompt = f"""You are an expert at reformulating questions for semantic vector search in RAG systems.
            Task: Turn the original user question into:
            - A clear, standalone, complete sentence or short paragraph
            - Using precise, technical terminology likely found in documents
            - Expanding abbreviations and adding context if it helps matching
            - Ideal length: 1–3 sentences
            Output ONLY the rewritten version — no explanations, no quotes, nothing else.

            Original question: {query}
            Rewritten:"""
                
            rewrite_query = llm.invoke(rewrite_prompt).strip()
            print("Rewritten query:",rewrite_query)

            cached = get_cached_response(rewrite_query)
            if cached:
                response = cached
            else:

                # 2. Generate HyDE answer based on the rewritten question
                hyde_prompt = f"""You are a world-class technical expert on LLMs and RAG.
                Write a concise but detailed hypothetical answer (3–6 sentences) to the following question.
                Write it in the style of a clear textbook or technical paper section.
                Be specific, use proper terminology, and explain concepts naturally.
                Do NOT say "I don't know" or refuse.

                Question: {rewrite_query}
                Hypothetical answer:"""

                hyde_answer = llm.invoke(hyde_prompt).strip()
                #print("\nGenerated HyDE answer:\n", hyde_answer, "\n")


                def hybrid_retrieve(query,rewrite_query, hyde_answer, k_vector=12, k_bm25=12, final_k=4):
                    # Vector retrieval (your existing)
                    query_vector = embeddings.embed_query(hyde_answer)  # returns list[float]
                    query_vector = np.array([query_vector]).astype("float32")  # shape (1, dim)
                    _, indices = index.search(query_vector, k_vector)
                    vector_results = [ texts[idx] for idx in indices[0] if idx < len(texts)]

                    # BM25 keyword retrieval
                    tokenized_query = rewrite_query.lower().split()
                    bm25_scores = bm25.get_scores(tokenized_query)
                    bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k_bm25]
                    bm25_results = [texts[i] for i in bm25_indices]

                    # Change how you create the combined list
                    combined = [{"text": chunk} for chunk in dict.fromkeys(vector_results + bm25_results)]

                    # Then
                    if combined :
                        rerank_request = RerankRequest(query=query, passages=combined)
                        reranked = reranker.rerank(rerank_request)
                        final_chunks = [item["text"] for item in reranked[:final_k]]
                    else:
                        final_chunks = []

                    return final_chunks
                
                final_chunks = hybrid_retrieve(query,rewrite_query, hyde_answer)

                # Debug print
                print("\nFinal chunks after hybrid + reranking:")
                for i, chunk in enumerate(final_chunks, 1):
                    print(f"{i}. {chunk[:200]}...")

                # Build proper context with IDs
                context = "\n\n".join(
                    f"[Chunk {i}] {chunk}" for i, chunk in enumerate(final_chunks)
                )
                        # 6. Final prompt — usually better to ask the original question
                prompt = f'''Kindly act as a helper, and answer the question using only the given context.
                Each paragraph in the context starts with a chunk ID.

                If you don't have enough information, say 'I don't know'.

                Context:
                {context}

                Question: {query}          # ← original question is usually better here

                Answer:
                - Provide a clear, concise answer
                - Cite sources using the chunk IDs (e.g., [Chunk 0])
                '''
                final_response = llm.invoke(prompt, max_tokens=600)

                is_good, reason = is_answer_good(query, final_response, context)

                retry_count = 0
                while not is_good and retry_count < 1:  # limit to 1 retry for now
                    retry_count += 1
                    print(f"Retry {retry_count} triggered: {reason}")

                    final_chunks = hybrid_retrieve(query, query, hyde_answer, k_vector=20)
                    context = "\n\n".join(f"[Chunk {i}] {c}" for i, c in enumerate(final_chunks))

                    retry_prompt = f"""Previous answer was not good enough: {reason}
                        Use this improved context to give a better answer.

                        Context:
                        {context}

                        Question: {query}

                        Answer:"""

                    final_response = llm.invoke(retry_prompt)

                    is_good, reason = is_answer_good(query, final_response, context)

                # After loop: final save only once
                if final_response and "don't know" not in final_response.lower():
                    save_to_cache(query, final_response)
                    if 'rewrite_query' in locals():
                        save_to_cache(rewrite_query, final_response)
                else:
                    final_response = "I don't know..."

    print(f"Response:\n{final_response}")
    return final_response

st.title("My RAG Agent Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about RAG, LLMs, or my documents..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = run_rag_pipeline(user_input)
                st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})