"""
RAG Engine for DocGenius.
Provides vector embedding with Gemini, FAISS indexing with cosine similarity,
metadata mapping, thresholded retrieval, retrieval context expansion (neighbor stitching),
grounded QA, and source citations.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import faiss
import google.generativeai as genai
from core.chunking import DocumentChunk

DEFAULT_GEN_MODEL = "models/gemini-flash-lite-latest"
FALLBACK_GEN_MODELS = [
    "models/gemini-flash-lite-latest",
    "models/gemini-3.8-flash",
    "models/gemini-flash-latest",
    "models/gemini-pro-latest"
]
DEFAULT_EMBED_MODEL = "models/gemini-embedding-001"
STRICT_REFUSAL_PHRASE = (
    "I couldn't find sufficient information in the uploaded document(s) to answer this question reliably."
)


@dataclass
class SearchResult:
    """Represents a retrieved chunk with similarity score, rank, and optional expanded context."""
    chunk: DocumentChunk
    similarity_score: float  # Cosine similarity in [-1, 1], typically [0, 1]
    rank: int
    expanded_text: Optional[str] = None  # Text after neighbor-chunk context expansion
    is_expanded: bool = False
    is_ocr: bool = False
    neighbor_pages: List[int] = field(default_factory=list)


class RAGEngine:
    """
    Manages vector indexing, semantic search, neighbor context expansion,
    and grounded response generation with resilient model fallbacks.
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
        self.last_diagnostics: Dict[str, Any] = {}

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
        self.last_diagnostics = {}

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

    def expand_chunk_context(
        self,
        chunk: DocumentChunk,
        window_size: int = 1
    ) -> Tuple[str, List[int]]:
        """
        Retrieval Context Expansion:
        Retrieves adjacent neighboring chunks (index - window to index + window)
        from the SAME document to ensure headings and explanations remain united.
        """
        doc_chunks = [c for c in self.chunks if c.doc_name == chunk.doc_name]
        if not doc_chunks:
            return chunk.text, [chunk.page_number]

        # Find position of this chunk within doc_chunks
        pos = -1
        for i, c in enumerate(doc_chunks):
            if c.chunk_id == chunk.chunk_id:
                pos = i
                break

        if pos == -1:
            return chunk.text, [chunk.page_number]

        start_pos = max(0, pos - window_size)
        end_pos = min(len(doc_chunks), pos + window_size + 1)

        stitched_parts: List[str] = []
        pages_included: List[int] = []

        for p in range(start_pos, end_pos):
            c_part = doc_chunks[p]
            stitched_parts.append(c_part.text)
            for pg in (c_part.spanned_pages or [c_part.page_number]):
                if pg not in pages_included:
                    pages_included.append(pg)

        expanded_text = "\n\n".join(stitched_parts)
        return expanded_text, pages_included

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.25,
        expand_context: bool = True,
        filter_doc: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Performs semantic retrieval for a user query with optional neighboring context expansion.
        Saves full diagnostics for developer inspection.
        """
        if not self.is_indexed() or not query.strip():
            self.last_diagnostics = {"status": "empty_or_no_query"}
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

        fetch_k = min(len(self.chunks), max(top_k * 3, 10))
        distances, indices = self.index.search(q_vec, fetch_k)

        candidates: List[Dict[str, Any]] = []
        accepted_results: List[SearchResult] = []
        seen_chunk_ids: set = set()
        rank = 1

        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            sim_score = float(score)
            chunk = self.chunks[idx]

            if filter_doc and chunk.doc_name != filter_doc:
                continue

            passed_threshold = (sim_score >= similarity_threshold)

            candidates.append({
                "chunk_id": chunk.chunk_id,
                "doc_name": chunk.doc_name,
                "page": chunk.page_number,
                "score": round(sim_score, 4),
                "passed_threshold": passed_threshold,
                "snippet": chunk.text[:100]
            })

            if not passed_threshold and len(accepted_results) >= 1:
                continue

            if chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.chunk_id)

            # Context Expansion: If chunk is short (<200 chars) or expand_context is requested
            expanded_text = chunk.text
            neighbor_pages = list(chunk.spanned_pages or [chunk.page_number])
            is_expanded = False

            if expand_context or len(chunk.text) < 200:
                expanded_text, neighbor_pages = self.expand_chunk_context(chunk, window_size=1)
                is_expanded = (expanded_text != chunk.text)

            accepted_results.append(SearchResult(
                chunk=chunk,
                similarity_score=round(sim_score, 4),
                rank=rank,
                expanded_text=expanded_text,
                is_expanded=is_expanded,
                is_ocr=getattr(chunk, "is_ocr", False),
                neighbor_pages=neighbor_pages
            ))
            rank += 1

            if len(accepted_results) >= top_k:
                break

        # Save diagnostics for debug view
        self.last_diagnostics = {
            "query": query,
            "threshold": similarity_threshold,
            "top_k": top_k,
            "expand_context": expand_context,
            "total_candidates_examined": len(candidates),
            "candidates": candidates,
            "accepted_count": len(accepted_results)
        }

        return accepted_results

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_results: List[SearchResult],
        chat_history: Optional[List[Dict[str, str]]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates an answer strictly grounded in the retrieved chunks.
        Applies resilient fallback across Gemini models if quota limits occur.
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

        context_blocks = []
        sources_meta = []

        for r in retrieved_results:
            c = r.chunk
            use_text = r.expanded_text if r.expanded_text else c.text
            pages_str = ", ".join(str(p) for p in r.neighbor_pages) if r.neighbor_pages else str(c.page_number)

            header = f"[Source: {c.doc_name} | Page(s): {pages_str} | Relevance: {r.similarity_score:.2f}]"
            context_blocks.append(f"{header}\n{use_text}")
            sources_meta.append({
                "doc_name": c.doc_name,
                "page_number": c.page_number,
                "pages_str": pages_str,
                "snippet": c.text,
                "expanded_snippet": use_text if r.is_expanded else None,
                "similarity_score": r.similarity_score,
                "chunk_id": c.chunk_id,
                "is_expanded": r.is_expanded,
                "is_ocr": getattr(c, "is_ocr", False)
            })

        context_str = "\n\n---\n\n".join(context_blocks)

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

        # Model Fallback Chain
        candidate_models = []
        if model_name:
            candidate_models.append(model_name)
        if self.gen_model not in candidate_models:
            candidate_models.append(self.gen_model)
        for fb in FALLBACK_GEN_MODELS:
            if fb not in candidate_models:
                candidate_models.append(fb)

        answer_text = ""
        last_error = ""
        successful_model = None

        for m_target in candidate_models:
            try:
                model = genai.GenerativeModel(m_target)
                response = model.generate_content(prompt, request_options={"timeout": 20})
                answer_text = response.text.strip()
                successful_model = m_target
                break
            except Exception as e:
                last_error = str(e)
                continue

        if not answer_text:
            answer_text = f"Error generating response from Gemini API: {last_error}"

        is_refusal = (STRICT_REFUSAL_PHRASE in answer_text)

        return {
            "answer": answer_text,
            "sources": sources_meta,
            "is_grounded": not is_refusal,
            "context_used": context_str,
            "model_used": successful_model
        }
