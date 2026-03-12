"""
RAG Evaluation Module.

Measures three key metrics:
    1. Retrieval Recall — did the correct source appear in search results?
    2. Answer Correctness — does the answer contain expected key facts?
    3. Faithfulness — is the answer grounded in the context (no hallucination)?

Usage:
    from src.evaluation.evaluator import RAGEvaluator

    evaluator = RAGEvaluator()
    report = evaluator.run_evaluation()
    evaluator.print_report(report)
"""

import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.retrieval.vector_store import VectorStore
from src.chains.rag_chain import RAGChain
from src.evaluation.test_dataset import EVAL_DATASET, SAMPLE_LEASE_TEXT


class RAGEvaluator:
    """Evaluate RAG pipeline quality using curated test data."""

    def __init__(self):
        self.parser = DocumentParser()
        self.chunker = TextChunker()
        self.vector_store = VectorStore()
        self.rag_chain = RAGChain()

    def setup_eval_data(self) -> int:
        """
        Create and ingest the evaluation document.
        Returns number of chunks created.
        """
        # Write eval document
        eval_path = Path("data/sample_docs/eval_lease.txt")
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(SAMPLE_LEASE_TEXT, encoding="utf-8")

        # Remove existing eval data to start fresh
        if self.vector_store.document_exists("eval_lease.txt"):
            self.vector_store.delete_document("eval_lease.txt")

        # Parse, chunk, store
        documents = self.parser.parse(str(eval_path))
        chunks = self.chunker.chunk(documents)
        num_stored = self.vector_store.add_chunks(
            chunks,
            doc_id="eval_lease.txt",
            collection_name="Evaluation",
        )

        return num_stored

    def evaluate_retrieval(self, question: str, expected_source: str, k: int = 5) -> dict:
        """
        Evaluate whether retrieval finds the right document.

        Returns:
            Dict with recall score and details.
        """
        results = self.vector_store.search(
            query=question,
            k=k,
            source_filters=None,  # Search everything
        )

        # Check if expected source appears in results
        retrieved_sources = [r["metadata"].get("source") for r in results]
        source_found = expected_source in retrieved_sources

        # Check rank — where does the correct source first appear?
        rank = None
        if source_found:
            rank = retrieved_sources.index(expected_source) + 1

        # Top result relevance score
        top_score = results[0]["score"] if results else 0

        return {
            "recall": 1.0 if source_found else 0.0,
            "rank": rank,
            "top_score": round(top_score, 3),
            "num_results": len(results),
            "retrieved_sources": retrieved_sources,
        }

    def evaluate_answer(self, answer: str, expected_keywords: list[str]) -> dict:
        """
        Evaluate whether the answer contains expected key facts.

        Returns:
            Dict with correctness score and details.
        """
        answer_lower = answer.lower()

        found_keywords = []
        missing_keywords = []

        for keyword in expected_keywords:
            if keyword.lower() in answer_lower:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)

        # Score = fraction of expected keywords found
        correctness = len(found_keywords) / len(expected_keywords) if expected_keywords else 0

        return {
            "correctness": round(correctness, 3),
            "found_keywords": found_keywords,
            "missing_keywords": missing_keywords,
            "total_expected": len(expected_keywords),
        }

    def evaluate_faithfulness(self, answer: str, sources: list[dict]) -> dict:
        """
        Basic faithfulness check — does the answer reference things
        that appear in the source text?

        This is a simple keyword overlap check. For production,
        you'd use LLM-as-Judge or RAGAS.

        Returns:
            Dict with faithfulness score and details.
        """
        if not sources:
            return {"faithfulness": 0.0, "detail": "No sources retrieved"}

        # Combine all source text
        source_text = " ".join(s.get("text_preview", "") for s in sources).lower()

        # Split answer into meaningful words (5+ chars to avoid noise)
        answer_words = set(
            word.strip(".,!?;:'\"()[]")
            for word in answer.lower().split()
            if len(word.strip(".,!?;:'\"()[]")) >= 5
        )

        if not answer_words:
            return {"faithfulness": 1.0, "detail": "Answer too short to evaluate"}

        # Count how many answer words appear in source text
        grounded_words = [w for w in answer_words if w in source_text]
        faithfulness = len(grounded_words) / len(answer_words)

        return {
            "faithfulness": round(faithfulness, 3),
            "grounded_words": len(grounded_words),
            "total_words": len(answer_words),
        }

    def run_single_eval(self, test_case: dict, k: int = 5) -> dict:
        """
        Run evaluation for a single test case.

        Returns:
            Dict with all metrics for this test case.
        """
        question = test_case["question"]
        expected_source = test_case["expected_source"]
        expected_keywords = test_case["expected_answer_keywords"]

        # Evaluate retrieval
        retrieval_result = self.evaluate_retrieval(question, expected_source, k=k)

        # Get RAG answer
        rag_result = self.rag_chain.ask(
            question=question,
            k=k,
            session_id=f"eval_{test_case['id']}",
            source_filters=None,
        )

        # Evaluate answer correctness
        answer_result = self.evaluate_answer(rag_result["answer"], expected_keywords)

        # Evaluate faithfulness
        faith_result = self.evaluate_faithfulness(rag_result["answer"], rag_result["sources"])

        # Clear eval session memory
        self.rag_chain.clear_memory(f"eval_{test_case['id']}")

        return {
            "id": test_case["id"],
            "question": question,
            "difficulty": test_case["difficulty"],
            "category": test_case["category"],
            "answer": rag_result["answer"],
            "retrieval": retrieval_result,
            "correctness": answer_result,
            "faithfulness": faith_result,
        }

    def run_evaluation(self, k: int = 5, delay: float = 2.0) -> dict:
        """
        Run full evaluation across all test cases.

        Args:
            k: Number of chunks to retrieve per question.
            delay: Seconds between API calls (rate limit protection).

        Returns:
            Full evaluation report with per-case and aggregate scores.
        """
        print("Setting up evaluation data...")
        num_chunks = self.setup_eval_data()
        print(f"  Stored {num_chunks} chunks for evaluation.\n")

        results = []
        total = len(EVAL_DATASET)

        for i, test_case in enumerate(EVAL_DATASET, 1):
            print(f"[{i}/{total}] Evaluating: {test_case['id']} - {test_case['question'][:50]}...")

            try:
                result = self.run_single_eval(test_case, k=k)
                results.append(result)

                # Brief status
                r = result["retrieval"]["recall"]
                c = result["correctness"]["correctness"]
                f = result["faithfulness"]["faithfulness"]
                print(f"         Retrieval: {r:.0%} | Correctness: {c:.0%} | Faithfulness: {f:.0%}")

            except Exception as e:
                print(f"         ERROR: {str(e)}")
                results.append({
                    "id": test_case["id"],
                    "question": test_case["question"],
                    "difficulty": test_case["difficulty"],
                    "category": test_case["category"],
                    "answer": f"ERROR: {str(e)}",
                    "retrieval": {"recall": 0},
                    "correctness": {"correctness": 0},
                    "faithfulness": {"faithfulness": 0},
                })

            # Rate limit protection
            if i < total:
                time.sleep(delay)

        # Aggregate scores
        report = self._compute_aggregate(results)
        return report

    def _compute_aggregate(self, results: list[dict]) -> dict:
        """Compute aggregate metrics from individual results."""
        if not results:
            return {"error": "No results"}

        total = len(results)

        avg_retrieval = sum(r["retrieval"]["recall"] for r in results) / total
        avg_correctness = sum(r["correctness"]["correctness"] for r in results) / total
        avg_faithfulness = sum(r["faithfulness"]["faithfulness"] for r in results) / total

        # By difficulty
        by_difficulty = {}
        for diff in ["easy", "medium", "hard", "unanswerable"]:
            subset = [r for r in results if r["difficulty"] == diff]
            if subset:
                by_difficulty[diff] = {
                    "count": len(subset),
                    "avg_retrieval": round(sum(r["retrieval"]["recall"] for r in subset) / len(subset), 3),
                    "avg_correctness": round(sum(r["correctness"]["correctness"] for r in subset) / len(subset), 3),
                    "avg_faithfulness": round(sum(r["faithfulness"]["faithfulness"] for r in subset) / len(subset), 3),
                }

        # By category
        by_category = {}
        categories = set(r["category"] for r in results)
        for cat in categories:
            subset = [r for r in results if r["category"] == cat]
            if subset:
                by_category[cat] = {
                    "count": len(subset),
                    "avg_retrieval": round(sum(r["retrieval"]["recall"] for r in subset) / len(subset), 3),
                    "avg_correctness": round(sum(r["correctness"]["correctness"] for r in subset) / len(subset), 3),
                }

        return {
            "total_questions": total,
            "aggregate": {
                "retrieval_recall": round(avg_retrieval, 3),
                "answer_correctness": round(avg_correctness, 3),
                "faithfulness": round(avg_faithfulness, 3),
            },
            "by_difficulty": by_difficulty,
            "by_category": by_category,
            "individual_results": results,
        }

    @staticmethod
    def print_report(report: dict):
        """Pretty-print the evaluation report."""
        print("\n" + "=" * 70)
        print("RAG EVALUATION REPORT")
        print("=" * 70)

        agg = report["aggregate"]
        print(f"\n{'AGGREGATE SCORES':^70}")
        print(f"{'─' * 70}")
        print(f"  Retrieval Recall:    {agg['retrieval_recall']:.1%}")
        print(f"  Answer Correctness:  {agg['answer_correctness']:.1%}")
        print(f"  Faithfulness:        {agg['faithfulness']:.1%}")
        print(f"  Total Questions:     {report['total_questions']}")

        print(f"\n{'SCORES BY DIFFICULTY':^70}")
        print(f"{'─' * 70}")
        print(f"  {'Difficulty':<15} {'Count':<8} {'Retrieval':<12} {'Correctness':<14} {'Faithfulness'}")
        for diff, scores in report["by_difficulty"].items():
            print(
                f"  {diff:<15} {scores['count']:<8} "
                f"{scores['avg_retrieval']:<12.1%} "
                f"{scores['avg_correctness']:<14.1%} "
                f"{scores['avg_faithfulness']:.1%}"
            )

        print(f"\n{'SCORES BY CATEGORY':^70}")
        print(f"{'─' * 70}")
        print(f"  {'Category':<20} {'Count':<8} {'Retrieval':<12} {'Correctness'}")
        for cat, scores in report["by_category"].items():
            print(
                f"  {cat:<20} {scores['count']:<8} "
                f"{scores['avg_retrieval']:<12.1%} "
                f"{scores['avg_correctness']:.1%}"
            )

        print(f"\n{'INDIVIDUAL RESULTS':^70}")
        print(f"{'─' * 70}")
        for r in report["individual_results"]:
            status = "✅" if r["correctness"]["correctness"] >= 0.5 else "❌"
            print(
                f"  {status} {r['id']:<4} [{r['difficulty']:<12}] "
                f"R:{r['retrieval']['recall']:.0%} "
                f"C:{r['correctness']['correctness']:.0%} "
                f"F:{r['faithfulness']['faithfulness']:.0%} "
                f"| {r['question'][:45]}"
            )

            if r["correctness"].get("missing_keywords"):
                print(f"       Missing: {r['correctness']['missing_keywords']}")

        print(f"\n{'=' * 70}")
