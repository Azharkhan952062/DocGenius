"""
RAG Engine for DocGenius.
Provides vector embedding with Gemini, FAISS indexing with cosine similarity,
metadata mapping, thresholded retrieval, grounded QA, and source citations.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import faiss
import google.generativeai as genai
from core.chunking import DocumentChunk

DEFAULT_GEN_MODEL = "models/gemini-3.8-flash"
DEFAULT_EMBED_MODEL = "models/gemini-embedding-001"
STRICT_REFUSAL_PHRASE = (
    "I couldn't find sufficient information in the uploaded document(s) to answer this question reliably."
)


@dataclass
class SearchResult:
    """Represents a retrieved chunk with similarity score and metadata."""
    chunk: DocumentChunk
    similarity_score: float  # Cosine similarity in [-1, 1], typically [0, 1]
    rank: int


class RAGEngine:
    """
    Manages vector indexing, semantic search, and grounded response generation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        embed_model: str = DEFAULT_EMBED_MODEL,
        gen_model: str = DEFAULT_GEN_MODEL
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)

        self.embed_model = embed_model
        self.gen_model = gen_model
        self.chunks: List[DocumentChunk] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.embedding_dimension: Optional[int] = None

    def is_configured(self) -> bool:
        """Checks if the Gemini API key is configured."""
        return bool(self.api_key)

    def is_indexed(self) -> bool:
        """Checks if the vector index is populated."""
        return self.index is not None and self.index.ntotal > 0

    def clear(self):
        """Resets the vector index and stored chunks."""
        self.chunks = []
        self.index = None
        self.embedding_dimension = None

    def embed_texts(
        self,
        texts: List[str],
        task_type: str = "retrieval_document",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """
        Embeds a list of texts using Gemini's embedding model.
        Returns unit-normalized float32 numpy vectors.
        """
        if not self.is_configured():
            raise ValueError("GEMINI_API_KEY is not configured.")

        vectors: List[List[float]] = []
        total = len(texts)

        for idx, text in enumerate(texts):
            # Safe truncation for embed endpoint if needed
            safe_text = text[:4000] if len(text) > 4000 else text
            res = genai.embed_content(
                model=self.embed_model,
                content=safe_text,
                task_type=task_type
            )
            raw_emb = res["embedding"]
            vectors.append(raw_emb)

            if progress_callback:
                progress_callback(idx + 1, total)

        arr = np.array(vectors, dtype="float32")
        # Normalize to unit length for Cosine Similarity via Inner Product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = arr / norms
        return normalized

    def build_index(
        self,
        chunks: List[DocumentChunk],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> int:
        """
        Embeds all chunks and builds a FAISS IndexFlatIP.
        Maps each vector index position to its DocumentChunk.
        """
        if not chunks:
            self.clear()
            return 0

        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        embeddings = self.embed_texts(
            texts=texts,
            task_type="retrieval_document",
            progress_callback=progress_callback
        )

        dim = embeddings.shape[1]
        self.embedding_dimension = dim
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        return self.index.ntotal

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.25,
        filter_doc: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Performs semantic retrieval for a user query.
        Returns top-k results matching the similarity threshold.
        """
        if not self.is_indexed() or not query.strip():
            return []

        # Embed query
        res = genai.embed_content(
            model=self.embed_model,
            content=query.strip(),
            task_type="retrieval_query"
        )
        q_vec = np.array([res["embedding"]], dtype="float32")
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Search FAISS (fetch more if document filtering is requested)
        fetch_k = min(len(self.chunks), max(top_k * 3, 10))
        distances, indices = self.index.search(q_vec, fetch_k)

        results: List[SearchResult] = []
        rank = 1

        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            sim_score = float(score)

            chunk = self.chunks[idx]
            if filter_doc and chunk.doc_name != filter_doc:
                continue

            # Apply similarity threshold
            if sim_score < similarity_threshold and len(results) >= 1:
                # If we already have strong matches, skip weak ones
                continue

            results.append(SearchResult(
                chunk=chunk,
                similarity_score=round(sim_score, 4),
                rank=rank
            ))
            rank += 1

            if len(results) >= top_k:
                break

        return results

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_results: List[SearchResult],
        chat_history: Optional[List[Dict[str, str]]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates an answer strictly grounded in the retrieved chunks.
        If no relevant chunks or context is insufficient, returns the exact refusal.
        """
        if not self.is_configured():
            return {
                "answer": "Error: GEMINI_API_KEY is not configured.",
                "sources": [],
                "is_grounded": False
            }

        if not retrieved_results:
            return {
                "answer": STRICT_REFUSAL_PHRASE,
                "sources": [],
                "is_grounded": False
            }

        # Format context with citations
        context_blocks = []
        sources_meta = []

        for r in retrieved_results:
            c = r.chunk
            header = f"[Source: {c.doc_name} | Page {c.page_number} | Relevance: {r.similarity_score:.2f}]"
            context_blocks.append(f"{header}\n{c.text}")
            sources_meta.append({
                "doc_name": c.doc_name,
                "page_number": c.page_number,
                "snippet": c.text,
                "similarity_score": r.similarity_score,
                "chunk_id": c.chunk_id
            })

        context_str = "\n\n---\n\n".join(context_blocks)

        # Build prompt
        prompt = f"""You are DocGenius, an expert AI Document Assistant and RAG specialist.

CRITICAL INSTRUCTIONS:
1. Answer the user question based PRIMARILY AND STRICTLY on the Document Context provided below.
2. In your answer, cite your sources inline using [Document Name, Page X] format whenever stating facts.
3. If the context does not contain sufficient information to answer the question reliably, you MUST state EXACTLY:
"{STRICT_REFUSAL_PHRASE}"
4. Do NOT hallucinate, infer facts outside the context, or use outside training data for document-specific claims.
5. Provide a professional, well-formatted, and accurate response.

DOCUMENT CONTEXT:
{context_str}

USER QUESTION:
{query}

RESPONSE:"""

        target_model = model_name or self.gen_model

        try:
            model = genai.GenerativeModel(target_model)
            response = model.generate_content(prompt)
            answer_text = response.text.strip()
        except Exception as e:
            # Fallback if specific model encounters error
            try:
                fallback_model = "models/gemini-flash-latest"
                model = genai.GenerativeModel(fallback_model)
                response = model.generate_content(prompt)
                answer_text = response.text.strip()
            except Exception as e2:
                answer_text = f"Error generating response from Gemini API: {str(e2)}"

        is_refusal = (STRICT_REFUSAL_PHRASE in answer_text)

        return {
            "answer": answer_text,
            "sources": sources_meta,
            "is_grounded": not is_refusal,
            "context_used": context_str
        }
