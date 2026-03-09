"""
Multi-turn Conversation Demo.

Shows how the RAG chain remembers context across follow-up questions.

Run with: python notebooks/test_conversation.py
"""

import sys
sys.path.append(".")

from src.chains.rag_chain import RAGChain


def main():
    chain = RAGChain()
    session = "demo_session"

    print("=" * 60)
    print("MULTI-TURN CONVERSATION DEMO")
    print("=" * 60)
    print("\nMake sure you've already uploaded a document via /upload")
    print("or run test_rag_chain.py first to load the sample lease.\n")

    # Conversation flow — each question builds on the last
    questions = [
        "How much is the monthly rent?",
        "What happens if I pay late?",
        "How much notice do I need to give to terminate?",
        "What about the security deposit — will I get it back?",
    ]

    for i, q in enumerate(questions, start=1):
        print(f"{'─' * 60}")
        print(f"[Turn {i}] You: {q}")
        print(f"{'─' * 60}")

        result = chain.ask(q, k=3, session_id=session)

        print(f"\nAssistant: {result['answer']}")
        print(f"\n  Sources: {result['num_sources']} passages used")
        print()

    # Show the conversation history
    print("=" * 60)
    print("CONVERSATION HISTORY")
    print("=" * 60)
    history = chain.get_memory(session)
    for msg in history:
        role = "You" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"  {role}: {content}")

    # Cleanup
    chain.clear_memory(session)
    print(f"\nSession '{session}' cleared.")
    print("Demo complete!")


if __name__ == "__main__":
    main()
