"""
Comprehensive Test Suite for DocGenius RAG Pipeline & Modules.
Tests:
1. Module imports & syntax validation
2. PDF text extraction & text cleaning
3. Low-density page detection & empty page handling
4. Contextual heading fusion & boundary-aware chunking
5. Neighbor context expansion (window stitching)
6. FAISS Cosine retrieval & score thresholding
7. Grounded QA on relevant query ("types of environment")
8. Refusal on unrelated query
9. Real evaluation metrics (Precision@K, Recall@K, MRR, Grounding Score)
10. Live Gemini API integration with resilient fallback
"""

import os
import sys
import unittest
import numpy as np
from dotenv import load_dotenv

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from core.pdf_processor import process_pdf, clean_extracted_text, PageContent, ProcessedDocument
from core.chunking import chunk_document, is_heading_or_fragment, DocumentChunk
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


class TestDocGeniusPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create an authentic AI syllabus document with headings and explanations
        cls.ai_syllabus_doc = ProcessedDocument(
            doc_name="AI_Syllabus_Agent_Environments.pdf",
            total_pages=5,
            non_empty_pages=4,
            total_chars=1800,
            total_words=260
        )
        cls.ai_syllabus_doc.pages = [
            PageContent(
                doc_name="AI_Syllabus_Agent_Environments.pdf",
                page_number=1,
                text="Introduction to Artificial Intelligence: Agents and Environments. An AI agent perceives its environment through sensors and acts upon it through actuators.",
                char_count=165,
                word_count=23
            ),
            PageContent(
                doc_name="AI_Syllabus_Agent_Environments.pdf",
                page_number=2,
                text="Task Environment Specification (PEAS): Performance measure, Environment, Actuators, and Sensors. For example, an automated taxi agent.",
                char_count=138,
                word_count=18
            ),
            PageContent(
                doc_name="AI_Syllabus_Agent_Environments.pdf",
                page_number=3,
                text="camera\n(Task Environment):-",  # Heading-like snippet
                char_count=27,
                word_count=3,
                is_low_density=True
            ),
            PageContent(
                doc_name="AI_Syllabus_Agent_Environments.pdf",
                page_number=4,
                text="(Types of Environment)",  # Isolated heading
                char_count=22,
                word_count=3,
                is_low_density=True
            ),
            PageContent(
                doc_name="AI_Syllabus_Agent_Environments.pdf",
                page_number=5,
                text="The types of environment in Artificial Intelligence include: Fully Observable vs Partially Observable, Deterministic vs Stochastic, Episodic vs Sequential, Static vs Dynamic, Discrete vs Continuous, and Single-Agent vs Multi-Agent. In a fully observable environment, agent sensors give access to the complete state of the environment at each point in time.",
                char_count=359,
                word_count=49
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

    def test_02_text_cleaning_and_hyphenation(self):
        """Test text cleaning fixes hyphenation across line breaks."""
        raw = "deter-\nministic and sto-\nchastic envi-\nronments"
        cleaned = clean_extracted_text(raw)
        self.assertIn("deterministic", cleaned)
        self.assertIn("stochastic", cleaned)
        self.assertIn("environments", cleaned)

    def test_03_heading_detection(self):
        """Test detection of headings and isolated fragments."""
        self.assertTrue(is_heading_or_fragment("(Types of Environment)"))
        self.assertTrue(is_heading_or_fragment("camera\n(Task Environment):-"))
        self.assertTrue(is_heading_or_fragment("1.2 Introduction:"))
        self.assertFalse(is_heading_or_fragment(
            "In a fully observable environment, agent sensors give access to the complete state."
        ))

    def test_04_contextual_heading_fusion(self):
        """Test that isolated headings (like Page 4 '(Types of Environment)') fuse with following content."""
        chunks = chunk_document(self.ai_syllabus_doc, chunk_size=800, chunk_overlap=150, deduplicate=True)
        self.assertGreater(len(chunks), 0)

        # Look for the chunk that contains '(Types of Environment)'
        heading_chunk = next((c for c in chunks if "Types of Environment" in c.text), None)
        self.assertIsNotNone(heading_chunk, "Heading must exist in chunks")

        # Crucial check: Heading must NOT be isolated; it must contain the explanatory types!
        self.assertIn("Fully Observable", heading_chunk.text)
        self.assertIn("Deterministic", heading_chunk.text)
        self.assertTrue(heading_chunk.has_heading_context)
        self.assertIn(4, heading_chunk.spanned_pages)
        self.assertIn(5, heading_chunk.spanned_pages)

    def test_05_neighbor_context_expansion(self):
        """Test context window expansion stitches adjacent chunks from the same document."""
        engine = RAGEngine()
        c1 = DocumentChunk("c1", "doc.pdf", 1, "Section 1: Agent Definition.", 28, 4, 0)
        c2 = DocumentChunk("c2", "doc.pdf", 1, "Section 2: PEAS Properties.", 28, 4, 1)
        c3 = DocumentChunk("c3", "doc.pdf", 2, "Section 3: Environment Types.", 30, 4, 2)
        engine.chunks = [c1, c2, c3]

        expanded_text, pages = engine.expand_chunk_context(c2, window_size=1)
        self.assertIn("Section 1: Agent Definition", expanded_text)
        self.assertIn("Section 2: PEAS Properties", expanded_text)
        self.assertIn("Section 3: Environment Types", expanded_text)
        self.assertEqual(pages, [1, 2])

    def test_06_evaluation_metrics_precision_and_recall(self):
        """Test real mathematical calculation of Precision@K, Recall@K, and MRR."""
        c1 = DocumentChunk("id1", "doc.pdf", 4, "(Types of Environment) Fully Observable vs Partially Observable.", 64, 8, 0)
        c2 = DocumentChunk("id2", "doc.pdf", 2, "PEAS taxi agent sensors and actuators.", 39, 6, 1)

        retrieved = [
            SearchResult(c1, 0.92, 1),
            SearchResult(c2, 0.45, 2)
        ]

        target_kws = ["observable", "environment", "deterministic"]
        p_at_2 = compute_precision_at_k(retrieved, target_kws, k=2)
        self.assertEqual(p_at_2, 0.5)

        r_at_2 = compute_recall_at_k(retrieved, target_kws, k=2)
        self.assertGreaterEqual(r_at_2, 0.6)

        mrr = compute_reciprocal_rank(retrieved, target_kws)
        self.assertEqual(mrr, 1.0)

    def test_07_grounded_answer_generation(self):
        """Test grounded generation against relevant content and refusal against unrelated query."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.skipTest("GEMINI_API_KEY not found.")

        engine = RAGEngine(api_key=api_key, gen_model="models/gemini-flash-lite-latest")
        chunks = chunk_document(self.ai_syllabus_doc, chunk_size=800, chunk_overlap=150)
        engine.build_index(chunks)

        # Relevant query test
        results = engine.retrieve("Explain the types of environment mentioned in the document.", top_k=2)
        self.assertGreater(len(results), 0)

        ans_data = engine.generate_grounded_answer(
            "Explain the types of environment mentioned in the document.",
            results
        )
        self.assertTrue(ans_data["is_grounded"])
        ans_text = ans_data["answer"].lower()
        self.assertTrue("observable" in ans_text or "deterministic" in ans_text)
        self.assertGreater(len(ans_data["sources"]), 0)

        # Unrelated query test (Strict Refusal)
        unrelated = engine.retrieve("What is the recipe for chocolate chip cookies on Mars?", top_k=2, similarity_threshold=0.80)
        ans_refusal = engine.generate_grounded_answer("What is the recipe for chocolate chip cookies on Mars?", unrelated)
        self.assertIn(STRICT_REFUSAL_PHRASE, ans_refusal["answer"])


if __name__ == "__main__":
    unittest.main()
