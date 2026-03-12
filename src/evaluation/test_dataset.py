"""
Evaluation Test Dataset.

Curated Q&A pairs based on the sample lease agreement.
Each entry has:
    - question: What a user would ask
    - expected_answer_keywords: Key facts that MUST appear in a correct answer
    - expected_source: Which document should be retrieved
    - expected_page: Which page contains the answer
    - difficulty: easy/medium/hard

Usage:
    from src.evaluation.test_dataset import EVAL_DATASET, SAMPLE_LEASE_TEXT
"""

# The sample legal document used for evaluation
SAMPLE_LEASE_TEXT = """RESIDENTIAL LEASE AGREEMENT

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
"""

# Evaluation dataset — each entry tests a specific capability
EVAL_DATASET = [
    # ---- EASY: Direct fact extraction ----
    {
        "id": "E1",
        "question": "How much is the monthly rent?",
        "expected_answer_keywords": ["1,500", "1500", "GBP"],
        "expected_source": "eval_lease.txt",
        "difficulty": "easy",
        "category": "fact_extraction",
    },
    {
        "id": "E2",
        "question": "Who is the landlord?",
        "expected_answer_keywords": ["John Smith"],
        "expected_source": "eval_lease.txt",
        "difficulty": "easy",
        "category": "fact_extraction",
    },
    {
        "id": "E3",
        "question": "Where is the property located?",
        "expected_answer_keywords": ["45 Baker Street", "London"],
        "expected_source": "eval_lease.txt",
        "difficulty": "easy",
        "category": "fact_extraction",
    },
    {
        "id": "E4",
        "question": "How much is the security deposit?",
        "expected_answer_keywords": ["3,000", "3000", "two months"],
        "expected_source": "eval_lease.txt",
        "difficulty": "easy",
        "category": "fact_extraction",
    },

    # ---- MEDIUM: Requires understanding context ----
    {
        "id": "M1",
        "question": "What happens if I pay rent late?",
        "expected_answer_keywords": ["5%", "penalty", "7 days"],
        "expected_source": "eval_lease.txt",
        "difficulty": "medium",
        "category": "reasoning",
    },
    {
        "id": "M2",
        "question": "Can I have a dog in the apartment?",
        "expected_answer_keywords": ["pets", "not allowed", "written approval"],
        "expected_source": "eval_lease.txt",
        "difficulty": "medium",
        "category": "reasoning",
    },
    {
        "id": "M3",
        "question": "How do I renew the lease?",
        "expected_answer_keywords": ["mutual", "written", "60 days"],
        "expected_source": "eval_lease.txt",
        "difficulty": "medium",
        "category": "reasoning",
    },
    {
        "id": "M4",
        "question": "What repairs am I responsible for as a tenant?",
        "expected_answer_keywords": ["minor", "100", "clean"],
        "expected_source": "eval_lease.txt",
        "difficulty": "medium",
        "category": "reasoning",
    },

    # ---- HARD: Multi-section reasoning or negation ----
    {
        "id": "H1",
        "question": "Under what conditions can the landlord terminate the lease immediately?",
        "expected_answer_keywords": ["30", "consecutive", "pay rent"],
        "expected_source": "eval_lease.txt",
        "difficulty": "hard",
        "category": "multi_section",
    },
    {
        "id": "H2",
        "question": "If I terminate early without notice, what do I lose?",
        "expected_answer_keywords": ["security deposit", "forfeiture"],
        "expected_source": "eval_lease.txt",
        "difficulty": "hard",
        "category": "multi_section",
    },
    {
        "id": "H3",
        "question": "Am I allowed to sublet or make changes to the property?",
        "expected_answer_keywords": ["sublet", "written consent", "structural alterations"],
        "expected_source": "eval_lease.txt",
        "difficulty": "hard",
        "category": "multi_section",
    },

    # ---- UNANSWERABLE: Should say it can't find the answer ----
    {
        "id": "U1",
        "question": "What is the landlord's phone number?",
        "expected_answer_keywords": ["cannot find", "not", "no information"],
        "expected_source": "eval_lease.txt",
        "difficulty": "unanswerable",
        "category": "unanswerable",
    },
]
