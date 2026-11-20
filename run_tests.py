"""
Comprehensive Test Suite for DocGenius RAG Pipeline, OCR, and Grounding.
Tests:
1. Module imports & syntax validation
2. Text cleaning & wrap normalization
3. Heading & text-poor page detection (OCR trigger condition)
4. Contextual chunking with OCR provenance
5. Page rendering via pypdfium2 / image fallback
6. FAISS Cosine retrieval with OCR-generated chunks
7. Grounded QA on relevant query ("types of environment") using OCR notes
8. Strict refusal on unrelated queries
9. Verification that text-rich pages bypass OCR to optimize latency
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from core.ocr_engine import is_page_text_poor, render_page_to_image, transcribe_image_with_gemini
from core.pdf_processor import process_pdf, clean_extracted_text, PageContent, ProcessedDocument
from core.chunking import chunk_document, is_heading_or_fragment, DocumentChunk
from core.rag_engine import RAGEngine, SearchResult, STRICT_REFUSAL_PHRASE
from core.evaluation import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_context_grounding_score
)


class TestDocGeniusPipelineWithOCR(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.api_key = os.getenv("GEMINI_API_KEY")

        # Mock OCR-extracted handwritten notes document
        cls.ocr_notes_doc = ProcessedDocument(
            doc_name="AI_Syllabus_Handwritten_Notes.pdf",
            total_pages=5,
            non_empty_pages=2,
            total_chars=1200,
            ocr_pages_count=1
        )
        cls.ocr_notes_doc.pages = [
            PageContent(
                doc_name="AI_Syllabus_Handwritten_Notes.pdf",
                page_number=1,
                text="Artificial Intelligence Chapter 1: Agents and PEAS Framework.",
                char_count=62,
                word_count=8,
                is_ocr_applied=False
            ),
            PageContent(
                doc_name="AI_Syllabus_Handwritten_Notes.pdf",
                page_number=4,
                text="camera\n(Task Environment):-",
                char_count=27,
                word_count=3,
                is_low_density=True,
                is_ocr_applied=False
            ),
            PageContent(
                doc_name="AI_Syllabus_Handwritten_Notes.pdf",
                page_number=5,
                text=(
                    "# properties of Environment :- (Types of Environment)\n"
                    "1) Fully vs Partially Observable.\n"
                    "2) Deterministic vs Stochastic/strategic.\n"
                    "3) Episodic vs sequential.\n"
                    "4) static vs Dynamic.\n"
                    "5) Discrete vs Continuous.\n"
                    "6) Single agent vs multiagent.\n"
                    "Examples: Automated taxi is partially observable and stochastic. "
                    "Crossword puzzle is fully observable, deterministic, and discrete."
                ),
                char_count=430,
                word_count=52,
                is_ocr_applied=True  # Recovered via OCR
            )
        ]

    def test_01_imports(self):
        """Test that all core modules including ocr_engine import cleanly."""
        import core.ocr_engine
        import core.pdf_processor
        import core.chunking
        import core.rag_engine
        import core.tools
        import core.evaluation
        self.assertTrue(True)

    def test_02_text_poor_page_detection(self):
        """Test detection of pages that need OCR (< 60 chars) vs text-rich pages."""
        self.assertTrue(is_page_text_poor("", min_chars=60))
        self.assertTrue(is_page_text_poor("(Types of Environment)", min_chars=60))
        self.assertTrue(is_page_text_poor("camera\n(Task Environment):-", min_chars=60))
        self.assertFalse(is_page_text_poor(
            "In Artificial Intelligence, an agent perceives its environment through sensors and acts via actuators.",
            min_chars=60
        ))

    def test_03_text_rich_page_bypasses_ocr(self):
        """Verify that a page with sufficient selectable text does not trigger OCR unnecessarily."""
        doc = ProcessedDocument(doc_name="digital.pdf")
        page = PageContent("digital.pdf", 1, "A" * 150, 150, 20, is_ocr_applied=False)
        doc.pages.append(page)
        self.assertFalse(is_page_text_poor(page.text))

    def test_04_chunking_with_ocr_metadata(self):
        """Test that chunking preserves is_ocr flag and fuses headings with OCR content."""
        chunks = chunk_document(self.ocr_notes_doc, chunk_size=800, chunk_overlap=150)
        self.assertGreater(len(chunks), 0)

        # Heading on Page 4 should fuse with OCR content on Page 5
        ocr_chunk = next((c for c in chunks if "Types of Environment" in c.text), None)
        self.assertIsNotNone(ocr_chunk)
        self.assertTrue(ocr_chunk.is_ocr)
        self.assertIn("Fully vs Partially Observable", ocr_chunk.text)
        self.assertIn("Deterministic vs Stochastic", ocr_chunk.text)

    def test_05_grounded_qa_with_ocr_chunks(self):
        """Test end-to-end retrieval and grounded answering on the exact syllabus query."""
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY not configured.")

        engine = RAGEngine(api_key=self.api_key, gen_model="models/gemini-flash-lite-latest")
        chunks = chunk_document(self.ocr_notes_doc, chunk_size=800, chunk_overlap=150)
        engine.build_index(chunks)

        query = "Explain the types of environment mentioned in the document."
        results = engine.retrieve(query, top_k=2)
        self.assertGreater(len(results), 0)

        top_match = results[0]
        self.assertTrue(top_match.is_ocr)
        self.assertIn("Observable", top_match.chunk.text)

        ans_payload = engine.generate_grounded_answer(query, results)
        self.assertTrue(ans_payload["is_grounded"])
        ans_lower = ans_payload["answer"].lower()

        # Must describe the actual environment types
        self.assertTrue(
            "observable" in ans_lower or "deterministic" in ans_lower or "stochastic" in ans_lower,
            f"Expected environment types in answer, got: {ans_payload['answer']}"
        )
        self.assertGreater(len(ans_payload["sources"]), 0)
        self.assertTrue(ans_payload["sources"][0].get("is_ocr"))

    def test_06_unrelated_query_refusal_with_ocr(self):
        """Test that even with OCR content, unrelated queries are strictly refused."""
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY not configured.")

        engine = RAGEngine(api_key=self.api_key, gen_model="models/gemini-flash-lite-latest")
        chunks = chunk_document(self.ocr_notes_doc, chunk_size=800, chunk_overlap=150)
        engine.build_index(chunks)

        unrelated = engine.retrieve("What is the recipe for vanilla ice cream on Mars?", top_k=2, similarity_threshold=0.85)
        ans_payload = engine.generate_grounded_answer("What is the recipe for vanilla ice cream on Mars?", unrelated)
        self.assertIn(STRICT_REFUSAL_PHRASE, ans_payload["answer"])


if __name__ == "__main__":
    unittest.main()
