import cohere
import numpy as np
import os
from pypdf import PdfReader
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))


def parse_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_chunks(chunks: list) -> list:
    response = co.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document",
        embedding_types=["float"]
    )
    return response.embeddings.float


def embed_query(query: str) -> list:
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"]
    )
    return response.embeddings.float[0]


def cosine_similarity(vec_a, vec_b) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def vector_search(query_embedding, chunk_embeddings, chunks, top_k: int = 10) -> list:
    scores = [
        cosine_similarity(query_embedding, chunk_emb)
        for chunk_emb in chunk_embeddings
    ]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices]


def rerank_chunks(query: str, candidate_chunks: list, top_n: int = 3) -> list:
    results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=candidate_chunks,
        top_n=top_n
    )
    return [candidate_chunks[r.index] for r in results.results]


def generate_answer(query: str, context_chunks: list) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    
    # Search web as supplementary source
    web_context = web_search(query)
    
    if web_context:
        combined_context = f"DOCUMENT CONTEXT:\n{context}\n\nWEB SEARCH RESULTS:\n{web_context}"
    else:
        combined_context = context

    prompt = f"""You are an intelligent document assistant. Answer the question as accurately and helpfully as possible.

First determine what type of question this is:
- If it contains options labeled A), B), C), D) or similar → it is an MCQ. Evaluate each option carefully, eliminate wrong ones with explanation, and give the final answer.
- If it is an open question, explanation request, or summary → answer it fully and clearly in plain language without treating it as MCQ.
- If it asks to explain or describe multiple concepts → explain each one clearly and separately.

Never treat an explanation request as an MCQ even if it mentions multiple concepts.

For MCQ questions specifically:
1. Look at ALL options first before deciding
2. Evaluate each option technically — do NOT just pick the one most related to the context
3. Use your full knowledge of the subject to evaluate each option
4. Give the final answer with clear reasoning for why each other option is WRONG

Important: Just because the context mentions something related to an option does NOT mean that option is correct. Evaluate the technical accuracy of each option independently.

CONTEXT FROM DOCUMENT (use as background reference):
{combined_context}

NOTE: For MCQ questions, first check if the document context contains enough detail to answer confidently. If it does, use it as the primary source. If the document only partially covers the topic, combine it with the web search results and your technical knowledge to reason through the options correctly. Never pick an answer just because the document mentions something related to it.

QUESTION:
{query}

ANSWER:"""

    response = co.chat(
        model="command-r-plus-08-2024",
        message=prompt,
        temperature=0.2
    )
    return response.text


def run_rag_pipeline(query: str, chunks: list, chunk_embeddings: list) -> dict:
    query_embedding = embed_query(query)
    candidates = vector_search(query_embedding, chunk_embeddings, chunks, top_k=10)
    best_chunks = rerank_chunks(query, candidates, top_n=3)
    answer = generate_answer(query, best_chunks)
    return {
        "answer": answer,
        "sources": best_chunks
    }

def web_search(query: str) -> str:
    """Search the web using DuckDuckGo as a fallback source."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except:
        pass
    return ""