"""
Document Tools Module for DocGenius.
Implements real document workflows:
- search_document
- summarize_document (with hierarchical reduction for long docs)
- extract_key_points
- compare_documents
- answer_from_context
- intent_router for automated tool dispatching
"""

import re
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from core.pdf_processor import ProcessedDocument
from core.chunking import DocumentChunk
from core.rag_engine import (
    RAGEngine,
    SearchResult,
    DEFAULT_GEN_MODEL,
    FALLBACK_GEN_MODELS,
    STRICT_REFUSAL_PHRASE
)


class DocumentTools:
    """
    Executes core document tools over processed PDFs and the RAG index.
    """

    def __init__(self, rag_engine: RAGEngine, gen_model: str = DEFAULT_GEN_MODEL):
        self.rag = rag_engine
        self.gen_model = gen_model

    def _generate_with_fallback(self, prompt: str) -> str:
        """Executes generation with automatic model fallback to avoid quota errors."""
        candidate_models = [self.gen_model]
        for fb in FALLBACK_GEN_MODELS:
            if fb not in candidate_models:
                candidate_models.append(fb)

        last_err = ""
        for m_name in candidate_models:
            try:
                m = genai.GenerativeModel(m_name)
                res = m.generate_content(prompt, request_options={"timeout": 20})
                return res.text.strip()
            except Exception as e:
                last_err = str(e)
                continue
        raise RuntimeError(f"All Gemini models failed: {last_err}")

    def search_document(
        self,
        query: str,
        top_k: int = 5,
        filter_doc: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes semantic search against the FAISS vector index.
        Returns ranked chunk matches with complete source metadata.
        """
        if not self.rag.is_indexed():
            return {
                "status": "error",
                "message": "Vector index is empty. Please process a document first.",
                "results": []
            }

        matches = self.rag.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=0.15,
            filter_doc=filter_doc
        )

        formatted = []
        for m in matches:
            formatted.append({
                "rank": m.rank,
                "document": m.chunk.doc_name,
                "page": m.chunk.page_number,
                "similarity_score": m.similarity_score,
                "snippet": m.chunk.text,
                "chunk_id": m.chunk.chunk_id
            })

        return {
            "status": "success",
            "query": query,
            "total_matches": len(formatted),
            "results": formatted
        }

    def summarize_document(
        self,
        doc_name: str,
        processed_docs: List[ProcessedDocument],
        summary_type: str = "Executive Summary"
    ) -> Dict[str, Any]:
        """
        Summarizes a document using hierarchical map-reduce if length exceeds context boundaries.
        """
        target_doc = next((d for d in processed_docs if d.doc_name == doc_name), None)
        if not target_doc:
            if processed_docs:
                target_doc = processed_docs[0]
            else:
                return {"status": "error", "message": f"Document '{doc_name}' not found."}

        full_text = target_doc.get_full_text()
        if not full_text.strip():
            return {"status": "error", "message": "Document contains no extractable text to summarize."}

        # Long document map-reduce strategy
        max_chunk_words = 1500
        words = full_text.split()

        if len(words) <= max_chunk_words:
            prompt = f"""You are DocGenius, an expert AI document analyst.
Provide a high-quality, professional {summary_type} of the following document.

Document: '{target_doc.doc_name}' ({target_doc.total_pages} pages)
Content:
{full_text}

Requirements:
1. Executive Overview / Core Purpose.
2. Major Sections or Thematic Breakdown.
3. Crucial Findings, Decisions, or Data Points.
4. Final Conclusion or Recommendations.

Format with clear Markdown headers, bold highlights, and bullet points.
"""
            try:
                res_text = self._generate_with_fallback(prompt)
                return {
                    "status": "success",
                    "doc_name": target_doc.doc_name,
                    "summary": res_text,
                    "method": "Direct Full-Context Synthesis",
                    "pages_analyzed": target_doc.non_empty_pages
                }
            except Exception as e:
                return {"status": "error", "message": f"Summarization failed: {str(e)}"}
        else:
            # Multi-stage hierarchical summarization (Map-Reduce)
            chunk_segments = []
            for i in range(0, len(words), max_chunk_words):
                chunk_segments.append(" ".join(words[i:i + max_chunk_words]))

            intermediate_summaries = []
            for idx, segment in enumerate(chunk_segments[:5]):  # Process up to 5 large sections
                sec_prompt = f"Summarize key aspects of Section {idx + 1} of document '{target_doc.doc_name}':\n\n{segment}"
                try:
                    s_text = self._generate_with_fallback(sec_prompt)
                    intermediate_summaries.append(f"### Section {idx + 1} Insights\n{s_text}")
                except Exception:
                    continue

            combined_insights = "\n\n".join(intermediate_summaries)
            final_prompt = f"""You are DocGenius. Synthesize the section summaries into a cohesive, professional {summary_type} for '{target_doc.doc_name}':

{combined_insights}

Structure:
- Executive Summary
- Core Themes & Methodology
- Key Insights & Evidence
- Overall Conclusion
"""
            try:
                final_text = self._generate_with_fallback(final_prompt)
                return {
                    "status": "success",
                    "doc_name": target_doc.doc_name,
                    "summary": final_text,
                    "method": f"Hierarchical Map-Reduce ({len(chunk_segments)} sections)",
                    "pages_analyzed": target_doc.non_empty_pages
                }
            except Exception as e:
                return {"status": "error", "message": f"Hierarchical summarization failed: {str(e)}"}

    def extract_key_points(
        self,
        doc_name: str,
        processed_docs: List[ProcessedDocument]
    ) -> Dict[str, Any]:
        """
        Extracts key takeaways, findings, data points, and conclusions into structured points.
        """
        target_doc = next((d for d in processed_docs if d.doc_name == doc_name), None)
        if not target_doc and processed_docs:
            target_doc = processed_docs[0]
        if not target_doc:
            return {"status": "error", "message": "No document available for key points extraction."}

        full_text = target_doc.get_full_text()
        # Bound text length for prompt
        sample_text = full_text[:12000]

        prompt = f"""You are DocGenius, an expert document analyst.
Extract the most critical key points and takeaways from '{target_doc.doc_name}'.

Document Content:
{sample_text}

Provide:
1. Top 5-8 Key Takeaways (bullet points with concise bold headers).
2. Important Metrics, Dates, or Quantifiable Facts (if present).
3. Primary Arguments or Methodologies.
4. Actionable Conclusions.

Only include facts explicitly stated in the document.
"""
        try:
            res_text = self._generate_with_fallback(prompt)
            return {
                "status": "success",
                "doc_name": target_doc.doc_name,
                "key_points": res_text,
                "pages_reviewed": target_doc.non_empty_pages
            }
        except Exception as e:
            return {"status": "error", "message": f"Key points extraction failed: {str(e)}"}

    def compare_documents(
        self,
        doc_names: List[str],
        processed_docs: List[ProcessedDocument],
        comparison_focus: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compares two or more documents, highlighting agreements, differences, and unique findings.
        """
        matching_docs = [d for d in processed_docs if d.doc_name in doc_names]
        if len(matching_docs) < 2:
            return {
                "status": "error",
                "message": "At least two valid processed documents are required for comparison."
            }

        doc_excerpts = []
        for d in matching_docs[:3]:  # Compare up to 3 documents
            excerpt = d.get_full_text()[:6000]
            doc_excerpts.append(f"=== DOCUMENT: '{d.doc_name}' ({d.total_pages} pages) ===\n{excerpt}")

        comparison_context = "\n\n".join(doc_excerpts)
        focus_instruction = f"Focus particularly on: {comparison_focus}" if comparison_focus else "Provide a comprehensive comparative analysis."

        prompt = f"""You are DocGenius, an advanced comparative document analyst.
Compare and contrast the following documents:
{', '.join([d.doc_name for d in matching_docs])}

{focus_instruction}

DOCUMENTS CONTENT:
{comparison_context}

Please provide:
1. Executive Comparison Overview.
2. Direct Comparison Matrix (Areas of agreement vs. divergence).
3. Unique Insights per document (explicitly citing which document states what).
4. Synthesis / Comparative Conclusion.

Strictly ground all statements in the text provided.
"""
        try:
            res_text = self._generate_with_fallback(prompt)
            return {
                "status": "success",
                "documents_compared": [d.doc_name for d in matching_docs],
                "comparison": res_text
            }
        except Exception as e:
            return {"status": "error", "message": f"Document comparison failed: {str(e)}"}

    def answer_from_context(
        self,
        query: str,
        retrieved_results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Directly delegates grounded QA with source citations."""
        return self.rag.generate_grounded_answer(query, retrieved_results)

    def route_intent(self, user_query: str, available_doc_names: List[str]) -> Dict[str, Any]:
        """
        Determines user intent from natural language input.
        Returns the matched tool, relevant arguments, and explanation.
        """
        q = user_query.strip().lower()

        # Check for comparison
        if (
            ("compare" in q or "difference between" in q or "versus" in q or " vs " in q)
            and len(available_doc_names) >= 2
        ):
            return {
                "tool": "compare_documents",
                "args": {
                    "doc_names": available_doc_names[:2],
                    "comparison_focus": user_query
                },
                "confidence": 0.95
            }

        # Check for summary
        if any(w in q for w in ["summarize", "summary", "give me an overview", "brief of", "synopsis"]):
            target_doc = available_doc_names[0] if available_doc_names else ""
            for name in available_doc_names:
                if name.lower() in q:
                    target_doc = name
                    break
            return {
                "tool": "summarize_document",
                "args": {"doc_name": target_doc},
                "confidence": 0.95
            }

        # Check for key points
        if any(w in q for w in ["key points", "takeaways", "bullet points", "highlights", "main findings"]):
            target_doc = available_doc_names[0] if available_doc_names else ""
            for name in available_doc_names:
                if name.lower() in q:
                    target_doc = name
                    break
            return {
                "tool": "extract_key_points",
                "args": {"doc_name": target_doc},
                "confidence": 0.95
            }

        # Check for search
        if q.startswith("search ") or q.startswith("find ") or "look for " in q or "search for " in q:
            clean_search_query = re.sub(r"^(search|find|look for|search for)\s+", "", q)
            return {
                "tool": "search_document",
                "args": {"query": clean_search_query},
                "confidence": 0.90
            }

        # Default is grounded QA
        return {
            "tool": "answer_from_context",
            "args": {"query": user_query},
            "confidence": 0.90
        }
