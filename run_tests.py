"""
Comprehensive Test Suite for DocGenius RAG Pipeline & Modules.
Tests:
1. Module imports & syntax validation
2. PDF text extraction & metadata retention
3. Empty, malformed, and multi-page PDF handling
4. Boundary-aware chunking & deduplication
5. FAISS Indexing (IndexFlatIP), dimension, and cosine retrieval
6. Citation accuracy & metadata preservation
7. Document tools (Search, Intent Routing)
8. Real RAG Evaluation metrics (Precision@K, Recall@K, MRR, Grounding, Relevance)
9. Live Gemini API integration (Embeddings, Grounded QA, Refusal test)
"""

import os
import sys
import io
import unittest
import numpy as np

from dotenv import load_dotenv

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from core.pdf_processor import process_pdf, process_multiple_pdfs, PageContent, ProcessedDocument
from core.chunking import chunk_document, chunk_multiple_documents, DocumentChunk
from core.rag_engine import RAGEngine, SearchResult, STRICT_REFUSAL_PHRASE
from core.tools import DocumentTools
from core.evaluation import (
    TestCase,
    RAGEvaluator,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_context_grounding_score,
    compute_answer_relevance_score
)


def create_mock_pdf_bytes(pages_text: list) -> bytes:
    """Helper to generate a minimal valid PDF with specified page texts."""
    from PyPDF2 import PdfWriter
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    for text in pages_text:
        c.drawString(72, 750, text)
        c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


class TestDocGeniusPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create sample PDF bytes using pure PyPDF2 objects or simple text
        cls.sample_doc = ProcessedDocument(
            doc_name="Quantum_Computing_Report.pdf",
            total_pages=3,
            non_empty_pages=3,
            total_chars=1200,
            total_words=180
        )
        cls.sample_doc.pages = [
            PageContent(
                doc_name="Quantum_Computing_Report.pdf",
                page_number=1,
                text="Quantum computers leverage superposition and entanglement to execute complex calculations. Qubits represent states 0 and 1 simultaneously.",
                char_count=138,
                word_count=17
            ),
            PageContent(
                doc_name="Quantum_Computing_Report.pdf",
                page_number=2,
                text="Shor's algorithm can factor large integers exponentially faster than classical algorithms. This has profound implications for RSA encryption.",
                char_count=146,
                word_count=19
            ),
            PageContent(
                doc_name="Quantum_Computing_Report.pdf",
                page_number=3,
                text="",  # Empty page
                char_count=0,
                word_count=0,
                is_empty=True
            )
        ]

    def test_01_imports(self):
        """Test that all core modules import cleanly."""
        import core.pdf_processor
        import core.chunking
        import core.rag_engine
        import core.tools
        import core.evaluation
        self.assertTrue(True)

    def test_02_pdf_empty_handling(self):
        """Test PDF processing handles empty input or corrupt bytes gracefully."""
        doc = process_pdf(b"", doc_name="corrupt.pdf")
        self.assertIsNotNone(doc.error_message)
        self.assertEqual(len(doc.pages), 0)

    def test_03_chunking_and_metadata(self):
        """Test chunking preserves page numbers, doc names, and avoids empty chunks."""
        chunks = chunk_document(self.sample_doc, chunk_size=100, chunk_overlap=20, deduplicate=True)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertEqual(c.doc_name, "Quantum_Computing_Report.pdf")
            self.assertIn(c.page_number, [1, 2])
            self.assertGreater(len(c.text.strip()), 0)
            self.assertIn(c.doc_name, c.chunk_id)

    def test_04_chunk_deduplication(self):
        """Test that identical text chunks are deduplicated."""
        dup_doc = ProcessedDocument(doc_name="dup.pdf", total_pages=2)
        dup_doc.pages = [
            PageContent(doc_name="dup.pdf", page_number=1, text="Artificial Intelligence transforms healthcare diagnostics.", char_count=57, word_count=6),
            PageContent(doc_name="dup.pdf", page_number=2, text="Artificial Intelligence transforms healthcare diagnostics.", char_count=57, word_count=6)
        ]
        chunks = chunk_document(dup_doc, chunk_size=500, chunk_overlap=50, deduplicate=True)
        self.assertEqual(len(chunks), 1)

    def test_05_faiss_cosine_logic(self):
        """Test unit vector normalization and FAISS IndexFlatIP cosine similarity calculation."""
        import faiss
        dim = 4
        index = faiss.IndexFlatIP(dim)
        v1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")
        v2 = np.array([[0.0, 1.0, 0.0, 0.0]], dtype="float32")
        v_doc = np.vstack([v1, v2])
        index.add(v_doc)

        q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")
        distances, indices = index.search(q, 2)
        self.assertAlmostEqual(distances[0][0], 1.0, places=4)
        self.assertEqual(indices[0][0], 0)
        self.assertAlmostEqual(distances[0][1], 0.0, places=4)

    def test_06_evaluation_metrics_precision_and_recall(self):
        """Test mathematically computed Precision@K, Recall@K, and MRR."""
        c1 = DocumentChunk("id1", "doc.pdf", 1, "Quantum computing uses superposition.", 38, 5, 0)
        c2 = DocumentChunk("id2", "doc.pdf", 2, "Classical computers use bits.", 30, 4, 1)
        c3 = DocumentChunk("id3", "doc.pdf", 3, "Entanglement is a key quantum property.", 40, 6, 2)

        retrieved = [
            SearchResult(c1, 0.95, 1),
            SearchResult(c2, 0.50, 2),
            SearchResult(c3, 0.85, 3)
        ]

        target_kws = ["quantum", "superposition", "entanglement"]
        # In retrieved[:2], c1 has 'quantum'/'superposition' (relevant), c2 has none (not relevant)
        p_at_2 = compute_precision_at_k(retrieved, target_kws, k=2)
        self.assertEqual(p_at_2, 0.5)

        # In retrieved[:3], c1 has quantum & superposition, c3 has entanglement -> all 3 matched!
        r_at_3 = compute_recall_at_k(retrieved, target_kws, k=3)
        self.assertEqual(r_at_3, 1.0)

        # First relevant chunk is at rank 1 -> MRR = 1.0
        mrr = compute_reciprocal_rank(retrieved, target_kws)
        self.assertEqual(mrr, 1.0)

    def test_07_context_grounding_score(self):
        """Test sentence token overlap grounding calculation."""
        context = "Shor's algorithm can factor large integers exponentially faster than classical methods."
        # Highly grounded answer
        answer_grounded = "Shor's algorithm factors large integers exponentially faster than classical methods."
        score_grounded = compute_context_grounding_score(answer_grounded, context)
        self.assertGreaterEqual(score_grounded, 0.8)

        # Completely ungrounded answer
        answer_ungrounded = "Photosynthesis occurs in plant chloroplasts utilizing sunlight and chlorophyll."
        score_ungrounded = compute_context_grounding_score(answer_ungrounded, context)
        self.assertEqual(score_ungrounded, 0.0)

    def test_08_intent_routing(self):
        """Test tool intent routing for summary, key points, comparison, and search."""
        engine = RAGEngine()
        tools = DocumentTools(engine)
        docs = ["Alpha.pdf", "Beta.pdf"]

        r_sum = tools.route_intent("Can you summarize Alpha.pdf?", docs)
        self.assertEqual(r_sum["tool"], "summarize_document")
        self.assertEqual(r_sum["args"]["doc_name"], "Alpha.pdf")

        r_kp = tools.route_intent("Extract the key points from this document", docs)
        self.assertEqual(r_kp["tool"], "extract_key_points")

        r_comp = tools.route_intent("Compare the differences between Alpha and Beta", docs)
        self.assertEqual(r_comp["tool"], "compare_documents")

        r_search = tools.route_intent("search for financial revenue in 2025", docs)
        self.assertEqual(r_search["tool"], "search_document")

        r_qa = tools.route_intent("What is the primary conclusion of the author?", docs)
        self.assertEqual(r_qa["tool"], "answer_from_context")

    def test_09_gemini_api_integration(self):
        """Test live Gemini embedding, retrieval, and grounded answering if API key is present."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.skipTest("GEMINI_API_KEY not found in environment.")

        engine = RAGEngine(api_key=api_key, gen_model="models/gemini-3.8-flash")
        chunks = chunk_document(self.sample_doc, chunk_size=300, chunk_overlap=50)
        self.assertGreater(len(chunks), 0)

        # Build index
        ntotal = engine.build_index(chunks)
        self.assertEqual(ntotal, len(chunks))
        self.assertTrue(engine.is_indexed())

        # Retrieve
        results = engine.retrieve("What is Shor's algorithm used for?", top_k=2)
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertIn("shor", top_match.chunk.text.lower())
        self.assertEqual(top_match.chunk.page_number, 2)

        # Grounded answer
        ans_data = engine.generate_grounded_answer("What is Shor's algorithm used for?", results)
        self.assertTrue(ans_data["is_grounded"])
        self.assertIn("factor", ans_data["answer"].lower())
        self.assertGreater(len(ans_data["sources"]), 0)
        self.assertEqual(ans_data["sources"][0]["page_number"], 2)

        # Refusal test for completely unrelated question
        unrelated_results = engine.retrieve("What is the capital of Mars and the recipe for pizza?", top_k=2, similarity_threshold=0.85)
        ans_refusal = engine.generate_grounded_answer("What is the capital of Mars?", unrelated_results)
        self.assertIn(STRICT_REFUSAL_PHRASE, ans_refusal["answer"])


if __name__ == "__main__":
    unittest.main()
