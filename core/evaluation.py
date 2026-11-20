"""
RAG Evaluation Module for DocGenius.
Implements real, mathematically verifiable retrieval and generation evaluation metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Context Grounding / Faithfulness Score
- Answer Relevance Score
Includes a built-in benchmark test harness and customizable evaluation runner.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from core.rag_engine import RAGEngine, SearchResult


@dataclass
class TestCase:
    """Represents a benchmark question with ground truth criteria."""
    question: str
    expected_answer: str
    target_keywords: List[str]  # Key terms that must appear in relevant retrieved chunks
    target_page: Optional[int] = None
    target_doc: Optional[str] = None


@dataclass
class EvaluationResult:
    """Stores computed evaluation metrics for a single test case."""
    question: str
    expected_answer: str
    generated_answer: str
    retrieved_sources: List[Dict[str, Any]]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    grounding_score: float
    relevance_score: float
    k: int
    is_grounded: bool
    notes: str


def compute_precision_at_k(retrieved: List[SearchResult], target_keywords: List[str], k: int) -> float:
    """
    Computes Precision@K:
    Fraction of top-k retrieved chunks that contain at least one key target term.
    """
    if not retrieved or k <= 0:
        return 0.0

    eval_subset = retrieved[:k]
    relevant_count = 0

    lower_targets = [kw.lower() for kw in target_keywords]

    for item in eval_subset:
        chunk_text = item.chunk.text.lower()
        if any(kw in chunk_text for kw in lower_targets):
            relevant_count += 1

    return round(relevant_count / len(eval_subset), 4)


def compute_recall_at_k(retrieved: List[SearchResult], target_keywords: List[str], k: int) -> float:
    """
    Computes Recall@K:
    Fraction of unique target keywords that are captured within the top-k retrieved chunks.
    """
    if not target_keywords:
        return 1.0
    if not retrieved or k <= 0:
        return 0.0

    eval_subset = retrieved[:k]
    combined_text = " ".join([item.chunk.text.lower() for item in eval_subset])

    matched_keywords = sum(1 for kw in target_keywords if kw.lower() in combined_text)
    return round(matched_keywords / len(target_keywords), 4)


def compute_reciprocal_rank(retrieved: List[SearchResult], target_keywords: List[str]) -> float:
    """
    Computes Reciprocal Rank (RR):
    1 / rank of the first relevant chunk. Returns 0.0 if no retrieved chunk is relevant.
    """
    if not retrieved or not target_keywords:
        return 0.0

    lower_targets = [kw.lower() for kw in target_keywords]

    for rank_idx, item in enumerate(retrieved):
        chunk_text = item.chunk.text.lower()
        if any(kw in chunk_text for kw in lower_targets):
            return round(1.0 / (rank_idx + 1), 4)

    return 0.0


def compute_context_grounding_score(generated_answer: str, context: str) -> float:
    """
    Measures the grounding / faithfulness of the answer against the context.
    Calculates sentence-level token overlap between answer statements and context tokens.
    """
    if not generated_answer.strip() or not context.strip():
        return 0.0

    # Clean and tokenize context
    context_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", context.lower()))
    if not context_tokens:
        return 0.0

    # Split answer into sentences
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", generated_answer) if len(s.strip()) > 10]
    if not sentences:
        # Fallback to full text tokens
        ans_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", generated_answer.lower()))
        overlap = ans_tokens.intersection(context_tokens)
        return round(len(overlap) / max(len(ans_tokens), 1), 4)

    grounded_sentence_count = 0
    for sentence in sentences:
        s_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", sentence.lower()))
        if not s_tokens:
            continue
        overlap = s_tokens.intersection(context_tokens)
        # Sentence is considered grounded if >= 60% of informative tokens appear in context
        if (len(overlap) / len(s_tokens)) >= 0.60:
            grounded_sentence_count += 1

    return round(grounded_sentence_count / len(sentences), 4)


def compute_answer_relevance_score(generated_answer: str, question: str) -> float:
    """
    Computes lexical relevance score between question and generated answer.
    """
    q_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", question.lower()))
    # Remove common question words
    stop_words = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "does", "doesnt", "explain", "describe", "tell"}
    q_informative = q_tokens - stop_words

    if not q_informative:
        return 1.0

    ans_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", generated_answer.lower()))
    matched = q_informative.intersection(ans_tokens)

    return round(len(matched) / len(q_informative), 4)


class RAGEvaluator:
    """
    Runs systematic evaluations of the RAG pipeline using real metrics.
    """

    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine

    def evaluate_test_case(self, test_case: TestCase, top_k: int = 4) -> EvaluationResult:
        """
        Executes retrieval and answering for a single test case, computing all metrics.
        """
        # Step 1: Retrieval
        retrieved = self.rag.retrieve(query=test_case.question, top_k=top_k)

        # Step 2: Generation
        answer_payload = self.rag.generate_grounded_answer(
            query=test_case.question,
            retrieved_results=retrieved
        )
        gen_answer = answer_payload["answer"]
        context_used = answer_payload.get("context_used", "")

        # Step 3: Compute Real Metrics
        p_at_k = compute_precision_at_k(retrieved, test_case.target_keywords, k=top_k)
        r_at_k = compute_recall_at_k(retrieved, test_case.target_keywords, k=top_k)
        mrr = compute_reciprocal_rank(retrieved, test_case.target_keywords)
        grounding = compute_context_grounding_score(gen_answer, context_used)
        relevance = compute_answer_relevance_score(gen_answer, test_case.question)

        sources_meta = []
        for r in retrieved:
            sources_meta.append({
                "document": r.chunk.doc_name,
                "page": r.chunk.page_number,
                "similarity_score": r.similarity_score,
                "text_preview": r.chunk.text[:140] + "..." if len(r.chunk.text) > 140 else r.chunk.text
            })

        notes = []
        if p_at_k >= 0.75:
            notes.append("High retrieval precision.")
        elif p_at_k == 0.0:
            notes.append("No target keywords found in top-k chunks.")

        if grounding >= 0.70:
            notes.append("Strongly grounded in context.")
        else:
            notes.append("Moderate or low direct token overlap with context.")

        return EvaluationResult(
            question=test_case.question,
            expected_answer=test_case.expected_answer,
            generated_answer=gen_answer,
            retrieved_sources=sources_meta,
            precision_at_k=p_at_k,
            recall_at_k=r_at_k,
            reciprocal_rank=mrr,
            grounding_score=grounding,
            relevance_score=relevance,
            k=top_k,
            is_grounded=answer_payload["is_grounded"],
            notes=" ".join(notes)
        )

    def evaluate_suite(self, test_cases: List[TestCase], top_k: int = 4) -> Dict[str, Any]:
        """
        Runs evaluation across a suite of test cases and aggregates scores.
        """
        results: List[EvaluationResult] = []

        for tc in test_cases:
            res = self.evaluate_test_case(tc, top_k=top_k)
            results.append(res)

        n = len(results) if results else 1
        mean_p_at_k = round(sum(r.precision_at_k for r in results) / n, 4)
        mean_r_at_k = round(sum(r.recall_at_k for r in results) / n, 4)
        mrr_total = round(sum(r.reciprocal_rank for r in results) / n, 4)
        mean_grounding = round(sum(r.grounding_score for r in results) / n, 4)
        mean_relevance = round(sum(r.relevance_score for r in results) / n, 4)

        return {
            "summary": {
                "total_tests": len(results),
                "mean_precision_at_k": mean_p_at_k,
                "mean_recall_at_k": mean_r_at_k,
                "mean_reciprocal_rank": mrr_total,
                "mean_grounding_score": mean_grounding,
                "mean_relevance_score": mean_relevance,
                "k": top_k
            },
            "detailed_results": results
        }
