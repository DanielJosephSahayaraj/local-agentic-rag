from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
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
import streamlit as st
from rank_bm25 import BM25Okapi


# ── Embeddings ──
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = Ollama(
    model="llama3.1:8b", # Match the pulled name
    temperature=0.7,
    num_predict=100 
)

# ── Load document ──
try:
    loader = PyPDFLoader("https://arxiv.org/pdf/2005.11401.pdf")
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
except Exception as e:
    print("Failed to load PDF:", e)
    raise

# ── Split ──
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,       # ↑ increased
    chunk_overlap=200     # ↑ increased
)
chunks = text_splitter.split_documents(documents)
texts = [chunk.page_content for chunk in chunks]
print(f"Created {len(texts)} chunks")

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



# ── One-time BM25 index creation (do this after loading texts) ──
tokenized_texts = [text.lower().split() for text in texts]  # simple tokenization
bm25 = BM25Okapi(tokenized_texts)

reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

import numpy as np
from typing import Dict, Optional

# Global in-memory cache (query embedding → answer)
# You can later save this to disk with pickle/json
cache: Dict[tuple, str] = {}           # tuple because numpy arrays are not hashable
SIMILARITY_THRESHOLD = 0.95          # tune between 0.90–0.96

def get_cached_response(query: str) -> Optional[str]:
    """
    Returns cached answer if very similar query was asked before.
    Returns None if no good match.
    """
    # Embed current query
    q_vec = np.array(embeddings.embed_query(query))

    for cached_vec_tuple, cached_answer in cache.items():
        cached_vec = np.array(cached_vec_tuple)
        similarity = np.dot(q_vec, cached_vec) / (
            np.linalg.norm(q_vec) * np.linalg.norm(cached_vec) + 1e-10
        )
        if similarity >= SIMILARITY_THRESHOLD:
            print(f"Cache HIT! Similarity = {similarity:.4f}")
            return cached_answer

    print("Cache miss → running full pipeline")
    return None


def save_to_cache(query: str, answer: str):
    """
    Save the query embedding and its answer to cache.
    """
    q_vec = embeddings.embed_query(query)
    # Convert to tuple so it can be used as dict key
    cache[tuple(q_vec)] = answer
    print("Saved to cache")

    def is_answer_good(question: str, answer: str, context: str) -> tuple[bool, str]:
        """
        Ask the LLM to judge if its own answer is good.
        Returns (is_good: bool, reason: str)
        """
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
with open('texts.pkl', 'rb')as f:
    texts= pickle.load(f)


query = "How do you explain RAG in llm ?"

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
        prompt = f''' Kindly answer the question clearly and concisely.

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
            if not is_good:
                print(f"Retry triggered: {reason}")

                # Retry strategy: more candidates + original query for retrieval
                final_chunks = hybrid_retrieve(query, query, hyde_answer, k_vector=20)  # wider search
                context = "\n\n".join(f"[Chunk {i}] {c}" for i, c in enumerate(final_chunks))

                retry_prompt = f"""Previous answer was not good enough: {reason}
                        Use this improved context to give a better answer.

                        Context:
                        {context}

                        Question: {query}

                        Answer:"""

                final_response = llm.invoke(retry_prompt)
            
            if final_response and "don't know" not in final_response.lower():
                    save_to_cache(query, final_response)
                    save_to_cache(rewrite_query, final_response)
            else:
                response = "I don't know..."
        
    print(f"Response:\n{final_response}")



import streamlit as st

st.title("Streamlit Test Page")
st.write("If you see this, Streamlit is working!")

st.title("My RAG Agent Chat")
# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask me about RAG, LLMs, or anything in my documents..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run your RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Your full pipeline here (cache check + decision + retrieval + retry loop + generation)
            # For now, placeholder — replace with your actual code
            response = "This is where your final_response would appear..."

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


    from ragas.run_config import RunConfig
eval_llm_instance = Ollama(
    model="llama3.1:8b", 
    temperature=0,      # We want the judge to be consistent
    num_predict=1024    # Give the judge enough space to explain its reasoning
)

# 2. Second, wrap it so Ragas can "talk" to it
ragas_llm = LangchainLLMWrapper(eval_llm_instance)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

eval_samples = [
    {
        "question": "What is Retrieval-Augmented Generation?",
        "answer": "RAG stands for Retrieval-Augmented Generation. It's a model that combines pre-trained parametric and non-parametric memory for language generation ([Chunk 0]). The parametric memory is a pre-trained seq2seq model, while the non-parametric memory is a dense vector index of Wikipedia accessed with a pre-trained neural retriever.",   # ← run your RAG once and paste
        "contexts": [" 1. explore a general-purpose ﬁne-tuning recipe for retrieval-augmented generation(RAG) — models which combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG... 2. mechanisms may not be necessary for RAG.G Parameters Our RAG models contain the trainable parameters for the BERT-base query and document encoder of DPR, with 110M parameters each (although we do not... 3. Appendices for Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks A Implementation Details For Open-domain QA we report test numbers using 15 retrieved documents for RAG-Token models. Fo... 4. Table 1 shows results for RAG along with state-of-the-art models. On all four open-domain QA tasks, RAG sets a new state of the art (only on the T5-comparable split for TQA). RAG combines the generati..."],  # ← the retrieved chunks
        "ground_truth": "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval of relevant documents with generation to improve factual accuracy and reduce hallucinations in LLMs."
    },
    {
        "question": "What is the main contribution of the Transformer paper?",
        "answer": "I don't know. The context only mentions related papers and their authors, but there's no information about the main contribution of a Transformer paper.",
        "contexts": ["1. [7] Christopher Clark and Matt Gardner. Simple and Effective Multi-Paragraph Reading Comprehension. a Xiv:1710.10723 [cs], October 2017. URL http://arxiv.org/abs/1710.10723.arXiv: 1710.10723.[8] J... 2. GB of CPU memory for all of Wikipedia. After submission, We have ported our code to HuggingFace Transformers [66]3, which achieves equivalent performance to the previous version but is a cleaner and e... 3. anthology/P19-1346. [13] Angela Fan, Claire Gardent, Chloe Braud, and Antoine Bordes. Augmenting transformers with KNN-based composite memory, 2020. URL https://openreview.net/forum?id=H1gx1CNKPH.[1...4. Linguistics. doi: 10.18653/v1/P18-1082. URL https://www.aclweb.org/anthology/P18-1082. [12] Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. ELI5: Long form qu..."],
        "ground_truth": "The main contribution is the introduction of the Transformer architecture based solely on attention mechanisms, eliminating recurrence and convolutions."
    },
    # Add 3–8 more examples like this (the more the better, but 5 is already useful)
]

# Convert to RAGAS Dataset format
dataset = Dataset.from_list(eval_samples)

slow_config = RunConfig(
    timeout=160,      # Increase timeout to 60 seconds
    max_retries=10,  # Give it more chances if it fails
    max_workers=1    # STRICTLY process one job at a time
)
# ───────────────────────────────────────────────────────────────
# 3. Run evaluation (this uses your local LLM as judge)
# ───────────────────────────────────────────────────────────────
result = evaluate(
    dataset=dataset,
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config= slow_config
)

# Print scores (each metric gives a 0–1 score + average)
print(result)