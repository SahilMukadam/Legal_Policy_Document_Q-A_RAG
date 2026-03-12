"""
Run the RAG evaluation pipeline.

This script:
1. Loads the sample lease document into the vector store
2. Runs 12 test questions (easy/medium/hard/unanswerable)
3. Measures retrieval recall, answer correctness, and faithfulness
4. Prints a detailed evaluation report

Run with: python notebooks/run_evaluation.py

Note: Makes ~12 Gemini API calls. With gemini-2.0-flash free tier
(1500 req/day), this is well within limits. Takes ~1-2 minutes.
"""

import sys
sys.path.append(".")

from src.evaluation.evaluator import RAGEvaluator


def main():
    print("=" * 70)
    print("LEGAL DOCUMENT Q&A — RAG EVALUATION")
    print("=" * 70)
    print("\nThis will run 12 test questions against the sample lease.")
    print("Each question tests retrieval accuracy, answer correctness,")
    print("and faithfulness (no hallucination).\n")

    evaluator = RAGEvaluator()

    # Run full evaluation (2s delay between calls for rate limiting)
    report = evaluator.run_evaluation(k=5, delay=2.0)

    # Print report
    evaluator.print_report(report)

    # Summary verdict
    agg = report["aggregate"]
    print("\n📋 VERDICT:")

    if agg["retrieval_recall"] >= 0.9:
        print("  ✅ Retrieval is excellent — finding the right passages consistently.")
    elif agg["retrieval_recall"] >= 0.7:
        print("  ⚠️ Retrieval is good but could be improved (try different chunk sizes).")
    else:
        print("  ❌ Retrieval needs work — relevant passages not being found.")

    if agg["answer_correctness"] >= 0.8:
        print("  ✅ Answer quality is strong — key facts are being captured.")
    elif agg["answer_correctness"] >= 0.5:
        print("  ⚠️ Answer quality is moderate — some key facts missing.")
    else:
        print("  ❌ Answer quality needs improvement.")

    if agg["faithfulness"] >= 0.7:
        print("  ✅ Faithfulness is good — answers are grounded in source text.")
    else:
        print("  ⚠️ Faithfulness could be improved — check for hallucination.")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
