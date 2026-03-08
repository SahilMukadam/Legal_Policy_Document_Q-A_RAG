"""
Manual RAG test — see the full pipeline in action.

This script:
1. Creates a sample legal document
2. Parses and stores it in the vector DB
3. Asks questions and shows the LLM's cited answers

Run with: python notebooks/test_rag_chain.py
"""

import sys
sys.path.append(".")

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.retrieval.vector_store import VectorStore
from src.chains.rag_chain import RAGChain


def main():
    # Step 1: Create a sample legal document
    sample_doc = "data/sample_docs/sample_lease.txt"
    with open(sample_doc, "w", encoding="utf-8") as f:
        f.write("""RESIDENTIAL LEASE AGREEMENT

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
100 GBP and must keep the property clean and in good condition. The Tenant 
must report any maintenance issues to the Landlord within 48 hours.

Section 6: Termination
Either party may terminate this lease with 60 days written notice. 
Early termination by the Tenant without proper notice will result in 
forfeiture of the security deposit. The Landlord may terminate immediately 
if the Tenant fails to pay rent for more than 30 consecutive days.

Section 7: Restrictions
The Tenant shall not sublet the property without written consent from 
the Landlord. No pets are allowed without prior written approval. 
Smoking is prohibited inside the property. The Tenant must not make 
structural alterations without the Landlord's consent.
""")

    print("=" * 60)
    print("LEGAL DOCUMENT Q&A — RAG DEMO")
    print("=" * 60)

    # Step 2: Parse and store
    parser = DocumentParser()
    chunker = TextChunker()
    store = VectorStore()
    chain = RAGChain()

    print("\n📄 Parsing sample lease agreement...")
    documents = parser.parse(sample_doc)
    chunks = chunker.chunk(documents)
    num_stored = store.add_chunks(chunks, doc_id="sample_lease")

    print(f"   Extracted {len(documents)} page(s)")
    print(f"   Created {len(chunks)} chunks")
    print(f"   Stored {num_stored} chunks in vector DB")

    # Step 3: Ask questions
    questions = [
        "When is rent due and how much is it?",
        "What happens if I pay rent late?",
        "How much is the security deposit?",
        "Can I have a pet in the property?",
        "How do I terminate the lease early?",
    ]

    for q in questions:
        print(f"\n{'─' * 60}")
        print(f"❓ Question: {q}")
        print(f"{'─' * 60}")

        result = chain.ask(q, k=3)

        print(f"\n💡 Answer:\n{result['answer']}")
        print(f"\n📚 Sources used ({result['num_sources']}):")
        for src in result["sources"]:
            print(f"   • {src['source']}, Page {src['page']} "
                  f"(relevance: {src['relevance_score']})")

    print(f"\n{'=' * 60}")
    print("Demo complete! Your RAG pipeline is working end-to-end.")
    print("=" * 60)


if __name__ == "__main__":
    main()
