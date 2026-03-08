"""
RAG Chain Module.

The core of the application — connects retrieval (vector search) to the LLM
(Gemini/Claude) to answer questions about legal documents with citations.

How it works:
    1. User asks: "When is rent due?"
    2. We search the vector store for relevant chunks
    3. We build a prompt with the chunks as context
    4. We send the prompt to the LLM
    5. The LLM answers using ONLY the provided context
    6. We return the answer + source citations

Usage:
    from src.chains.rag_chain import RAGChain

    chain = RAGChain()
    result = chain.ask("When is rent due?")
    # Returns: {"answer": "...", "sources": [...], "query": "..."}
"""

from src.retrieval.vector_store import VectorStore
from src.llm_provider import get_llm


# System prompt that instructs the LLM how to behave
SYSTEM_PROMPT = """You are a precise legal document assistant. Your job is to answer 
questions about legal documents using ONLY the provided context passages.

RULES:
1. Answer based ONLY on the provided context. Never make up information.
2. If the context doesn't contain enough information to answer, say 
   "I cannot find this information in the uploaded documents."
3. After your answer, cite which source(s) you used in this format:
   [Source: filename, Page X]
4. Keep answers clear, concise, and professional.
5. If multiple context passages are relevant, synthesize them into one coherent answer.
6. Quote specific phrases from the documents when relevant, using quotation marks.
"""

QUERY_TEMPLATE = """Context passages from uploaded legal documents:

{context}

---

Question: {question}

Provide a clear answer based on the context above, with source citations."""


class RAGChain:
    """Retrieval-Augmented Generation chain for legal document Q&A."""

    def __init__(self):
        """Initialize the RAG chain with vector store and LLM."""
        self.vector_store = VectorStore()
        self.llm = get_llm()

    def _format_context(self, results: list[dict]) -> str:
        """
        Format retrieved chunks into a context string for the LLM prompt.

        Args:
            results: Search results from VectorStore.search()

        Returns:
            Formatted string with numbered passages and metadata.
        """
        context_parts = []
        for i, result in enumerate(results, start=1):
            source = result["metadata"].get("source", "Unknown")
            page = result["metadata"].get("page", "N/A")
            score = result.get("score", 0)

            context_parts.append(
                f"[Passage {i}] (Source: {source}, Page: {page}, "
                f"Relevance: {score:.2f})\n{result['text']}"
            )

        return "\n\n".join(context_parts)

    def _extract_sources(self, results: list[dict]) -> list[dict]:
        """
        Extract source information from search results for citation.

        Args:
            results: Search results from VectorStore.search()

        Returns:
            List of source dicts with filename, page, score, and text preview.
        """
        sources = []
        for result in results:
            sources.append({
                "source": result["metadata"].get("source", "Unknown"),
                "page": result["metadata"].get("page", "N/A"),
                "chunk_index": result["metadata"].get("chunk_index", "N/A"),
                "relevance_score": round(result.get("score", 0), 3),
                "text_preview": result["text"][:150] + "..."
                    if len(result["text"]) > 150
                    else result["text"],
            })
        return sources

    def ask(self, question: str, k: int = 5) -> dict:
        """
        Ask a question about the uploaded documents.

        Args:
            question: Natural language question.
            k: Number of context chunks to retrieve.

        Returns:
            Dict with:
                - "answer": The LLM's answer with citations
                - "sources": List of source passages used
                - "query": The original question
                - "num_sources": How many passages were retrieved
        """
        # Step 1: Retrieve relevant chunks
        results = self.vector_store.search(query=question, k=k)

        if not results:
            return {
                "answer": "No documents have been uploaded yet. "
                          "Please upload a legal document first.",
                "sources": [],
                "query": question,
                "num_sources": 0,
            }

        # Step 2: Format context from retrieved chunks
        context = self._format_context(results)

        # Step 3: Build the prompt
        user_message = QUERY_TEMPLATE.format(
            context=context,
            question=question,
        )

        # Step 4: Call the LLM
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", user_message),
        ]

        response = self.llm.invoke(messages)

        # Step 5: Extract sources for citation
        sources = self._extract_sources(results)

        return {
            "answer": response.content,
            "sources": sources,
            "query": question,
            "num_sources": len(sources),
        }
