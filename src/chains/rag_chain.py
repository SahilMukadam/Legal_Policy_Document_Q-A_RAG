"""
RAG Chain Module with Conversation Memory and Source Filtering.

Connects retrieval (vector search) to the LLM (Gemini/Claude) to answer
questions about legal documents with citations. Supports:
    - Multi-turn conversations (follow-up questions)
    - Source filtering (restrict answers to specific documents)

Usage:
    from src.chains.rag_chain import RAGChain

    chain = RAGChain()
    r1 = chain.ask("When is rent due?", session_id="user1")
    r2 = chain.ask("What about late fees?", session_id="user1")
    r3 = chain.ask("Summarize this", source_filter="lease.pdf")
"""

import time
from src.retrieval.vector_store import VectorStore
from src.llm_provider import get_llm


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
7. If the user asks a follow-up question, use the conversation history 
   to understand what they're referring to.
"""

QUERY_TEMPLATE = """Context passages from uploaded legal documents:

{context}

---

{history_section}

Current Question: {question}

Provide a clear answer based on the context above, with source citations."""


class RAGChain:
    """Retrieval-Augmented Generation chain with memory and source filtering."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = get_llm()
        self._memory: dict[str, list[dict]] = {}

    def _get_history(self, session_id: str) -> list[dict]:
        return self._memory.get(session_id, [])

    def _add_to_memory(self, session_id: str, role: str, content: str):
        if session_id not in self._memory:
            self._memory[session_id] = []

        self._memory[session_id].append({
            "role": role,
            "content": content,
        })

        if len(self._memory[session_id]) > 20:
            self._memory[session_id] = self._memory[session_id][-20:]

    def _format_history(self, session_id: str) -> str:
        history = self._get_history(session_id)
        if not history:
            return ""

        parts = ["Previous conversation:"]
        for msg in history:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            if msg["role"] == "assistant" and len(content) > 300:
                content = content[:300] + "..."
            parts.append(f"{prefix}: {content}")

        return "\n".join(parts)

    def _format_context(self, results: list[dict]) -> str:
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

    def _build_search_query(self, question: str, session_id: str) -> str:
        history = self._get_history(session_id)
        if not history:
            return question

        last_user_msgs = [m for m in history if m["role"] == "user"]
        if last_user_msgs:
            last_question = last_user_msgs[-1]["content"]
            return f"{last_question} {question}"

        return question

    def ask(
        self,
        question: str,
        k: int = 5,
        session_id: str = "default",
        source_filter: str | None = None,
    ) -> dict:
        """
        Ask a question about the uploaded documents.

        Args:
            question: Natural language question.
            k: Number of context chunks to retrieve.
            session_id: Conversation session ID for memory.
            source_filter: Optional filename to restrict to one document.

        Returns:
            Dict with answer, sources, query, session_id, and source_filter.
        """
        # Step 1: Retrieve relevant chunks
        search_query = self._build_search_query(question, session_id)
        results = self.vector_store.search(
            query=search_query,
            k=k,
            source_filter=source_filter,
        )

        if not results:
            msg = "No documents have been uploaded yet." if not source_filter else \
                  f"No relevant content found in '{source_filter}'."
            return {
                "answer": msg + " Please upload a legal document first.",
                "sources": [],
                "query": question,
                "num_sources": 0,
                "session_id": session_id,
                "source_filter": source_filter,
            }

        # Step 2: Format context and history
        context = self._format_context(results)
        history_text = self._format_history(session_id)
        history_section = history_text if history_text else "No previous conversation."

        # Step 3: Build the prompt
        user_message = QUERY_TEMPLATE.format(
            context=context,
            history_section=history_section,
            question=question,
        )

        # Step 4: Call the LLM (with retry for rate limits)
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", user_message),
        ]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 35
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise

        # Step 5: Store in memory
        self._add_to_memory(session_id, "user", question)
        self._add_to_memory(session_id, "assistant", response.content)

        # Step 6: Return result
        sources = self._extract_sources(results)

        return {
            "answer": response.content,
            "sources": sources,
            "query": question,
            "num_sources": len(sources),
            "session_id": session_id,
            "source_filter": source_filter,
        }

    def clear_memory(self, session_id: str = "default"):
        if session_id in self._memory:
            del self._memory[session_id]

    def get_memory(self, session_id: str = "default") -> list[dict]:
        return self._get_history(session_id)
