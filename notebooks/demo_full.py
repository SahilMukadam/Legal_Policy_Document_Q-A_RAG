"""
Full Demo Script — use this to record your demo GIF/video.

This script demonstrates the complete RAG pipeline:
1. Upload a legal document
2. Ask questions and get cited answers
3. Follow-up questions with memory
4. Search filtering by collection
5. Show evaluation results

Run with: python notebooks/demo_full.py

Tip for recording: Use a tool like ScreenToGif (Windows) or
Kap (Mac) to record your terminal while this runs.
"""

import sys
import time
sys.path.append(".")

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.retrieval.store_provider import get_vector_store
from src.chains.rag_chain import RAGChain


def slow_print(text, delay=0.02):
    """Print text character by character for demo effect."""
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()


def section(title):
    print(f"\n{'━' * 60}")
    slow_print(f"  {title}")
    print(f"{'━' * 60}")


def main():
    parser = DocumentParser()
    chunker = TextChunker()
    store = get_vector_store()
    chain = RAGChain()

    # ============================================================
    section("⚖️  LEGAL DOCUMENT Q&A — FULL DEMO")
    # ============================================================

    print("\nThis demo shows the complete RAG pipeline in action.\n")
    time.sleep(1)

    # ---- Step 1: Ingest Documents ----
    section("📄 STEP 1: Document Ingestion")

    # Create sample docs
    lease_text = """RESIDENTIAL LEASE AGREEMENT

Section 1: Parties and Property
This lease agreement is entered into between John Smith (Landlord) and 
Jane Doe (Tenant) for the property located at 45 Baker Street, London, 
NW1 6XE, United Kingdom.

Section 2: Term
The lease term begins on January 1, 2025 and ends on December 31, 2025.
The lease may be renewed for additional 12-month periods upon mutual 
written agreement at least 60 days before the expiration date.

Section 3: Rent
The monthly rent is 1,500 GBP, due on the first day of each month.
Late payments will incur a penalty of 5% of the monthly rent if not 
received within 7 days of the due date. Rent shall be paid via bank 
transfer to the Landlord's designated account.

Section 4: Security Deposit
A security deposit of 3,000 GBP (equivalent to two months' rent) is 
required upon signing. The deposit will be held in a government-approved 
tenancy deposit scheme. The deposit will be returned within 30 days of 
lease termination, minus any deductions for damages beyond normal wear 
and tear.

Section 5: Maintenance and Repairs
The Landlord is responsible for structural repairs, plumbing, electrical 
systems, and heating. The Tenant is responsible for minor repairs under 
100 GBP and must keep the property clean and in good condition.

Section 6: Termination
Either party may terminate this lease with 60 days written notice.
Early termination by the Tenant without proper notice will result in 
forfeiture of the security deposit.

Section 7: Restrictions
The Tenant shall not sublet the property without written consent.
No pets are allowed without prior written approval. Smoking is 
prohibited inside the property.
"""

    privacy_text = """PRIVACY POLICY - Effective January 1, 2025

1. Data Collection
We collect personal data including name, email address, phone number,
and payment information when you create an account or make a purchase.
We also collect usage data through cookies and analytics tools.

2. Data Usage
Your personal data is used to provide and improve our services,
process payments, send important notifications, and comply with 
legal obligations. We do not sell your personal data to third parties.

3. Data Retention
We retain your personal data for a maximum of 3 years after account
closure. You may request deletion at any time by contacting our 
Data Protection Officer at dpo@example.com.

4. Your Rights
Under GDPR, you have the right to access, rectify, and delete your
personal data. You may also request data portability or object to
certain types of processing.

5. Security
All data is encrypted at rest using AES-256 and in transit using TLS 1.3.
We conduct annual security audits and penetration testing.
"""

    import os
    os.makedirs("data/sample_docs", exist_ok=True)

    with open("data/sample_docs/demo_lease.txt", "w") as f:
        f.write(lease_text)
    with open("data/sample_docs/demo_privacy.txt", "w") as f:
        f.write(privacy_text)

    # Clean up existing
    if store.document_exists("demo_lease.txt"):
        store.delete_document("demo_lease.txt")
    if store.document_exists("demo_privacy.txt"):
        store.delete_document("demo_privacy.txt")

    # Parse and store
    print("Uploading: Lease Agreement → 'Real Estate' collection")
    docs1 = parser.parse("data/sample_docs/demo_lease.txt")
    chunks1 = chunker.chunk(docs1)
    n1 = store.add_chunks(chunks1, doc_id="demo_lease.txt", collection_name="Real Estate")
    print(f"  ✅ {len(docs1)} page(s), {n1} chunks stored\n")

    print("Uploading: Privacy Policy → 'Compliance' collection")
    docs2 = parser.parse("data/sample_docs/demo_privacy.txt")
    chunks2 = chunker.chunk(docs2)
    n2 = store.add_chunks(chunks2, doc_id="demo_privacy.txt", collection_name="Compliance")
    print(f"  ✅ {len(docs2)} page(s), {n2} chunks stored\n")

    # Show collections
    collections = store.list_collections()
    print("Collections:")
    for coll, docs in collections.items():
        print(f"  📁 {coll}: {[d['source'] for d in docs]}")

    chain.invalidate_caches()
    time.sleep(1)

    # ---- Step 2: Ask Questions ----
    section("💬 STEP 2: Ask Questions (with Citations)")

    session = "demo"

    questions = [
        ("How much is the monthly rent?", None),
        ("What happens if I pay late?", None),
        ("What data do you collect about me?", ["demo_privacy.txt"]),
    ]

    for q, filters in questions:
        print(f"\n❓ {q}")
        if filters:
            print(f"   (filtered to: {filters})")

        result = chain.ask(q, k=3, session_id=session, source_filters=filters)

        print(f"\n💡 {result['answer'][:300]}...")
        print(f"\n   📚 Sources: {result['num_sources']} passages")
        for s in result["sources"][:2]:
            print(f"      • {s['source']}, Page {s['page']} "
                  f"[{s['collection']}] (score: {s['relevance_score']})")

        time.sleep(2)

    # ---- Step 3: Follow-up ----
    section("🔄 STEP 3: Follow-up Question (Memory)")

    print("\n❓ What about the security deposit?")
    print("   (system remembers we were discussing the lease)\n")

    result = chain.ask(
        "What about the security deposit?",
        k=3, session_id=session,
    )
    print(f"💡 {result['answer'][:300]}...")
    time.sleep(2)

    # ---- Step 4: Stats ----
    section("📊 STEP 4: System Stats")

    stats = store.get_stats()
    cache_stats = chain.get_cache_stats()

    print(f"  Documents:   {stats['total_documents']}")
    print(f"  Collections: {stats['total_collections']}")
    print(f"  Chunks:      {stats['total_chunks']}")
    print(f"  Cache hits:  {cache_stats['answer_cache']['total_hits']} answers, "
          f"{cache_stats['search_cache']['total_hits']} searches")

    # ---- Done ----
    section("✅ DEMO COMPLETE")

    print("\nThis project demonstrates:")
    print("  • RAG pipeline (parse → chunk → embed → search → answer)")
    print("  • Hybrid search (semantic + BM25 with rank fusion)")
    print("  • Collection-based document management")
    print("  • Conversation memory for follow-up questions")
    print("  • Source citations for every answer")
    print("  • Response caching for performance")
    print("  • Swappable LLM and vector store backends")
    print("  • 60+ automated tests")
    print("  • Evaluation pipeline (retrieval, correctness, faithfulness)")
    print(f"\n{'━' * 60}")

    # Cleanup
    chain.clear_memory(session)


if __name__ == "__main__":
    main()
