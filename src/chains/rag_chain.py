"""
RAG Chain Module with Conversation Memory, Multi-Source Filtering,
Hybrid Search, and Response Caching.
"""

import time
from src.retrieval.store_provider import get_vector_store
from src.retrieval.hybrid_search import HybridSearch
from src.llm_provider import get_llm
from src.utils.cache import ResponseCache


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
    """RAG chain with memory, filtering, hybrid search, and caching."""

    def __init__(self):
        self.vector_store = get_vector_store()
        self.hybrid_search = HybridSearch(self.vector_store)
        self.llm = get_llm()
        self._memory: dict[str, list[dict]] = {}

        # Cache for search results (shorter TTL — documents might change)
        self.search_cache = ResponseCache(ttl_seconds=120, max_size=200)

        # Cache for RAG answers (longer TTL — same question = same answer)
        self.answer_cache = ResponseCache(ttl_seconds=300, max_size=100)

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
            collection = result["metadata"].get("collection", "")
            score = result.get("score", 0)
            search_type = result.get("search_type", "semantic")

            context_parts.append(
                f"[Passage {i}] (Source: {source}, Collection: {collection}, "
                f"Page: {page}, Relevance: {score:.4f}, Method: {search_type})\n{result['text']}"
            )

        return "\n\n".join(context_parts)

    def _extract_sources(self, results: list[dict]) -> list[dict]:
        sources = []
        for result in results:
            sources.append({
                "source": result["metadata"].get("source", "Unknown"),
                "page": result["metadata"].get("page", "N/A"),
                "collection": result["metadata"].get("collection", "General"),
                "chunk_index": result["metadata"].get("chunk_index", "N/A"),
                "relevance_score": round(result.get("score", 0), 4),
                "search_type": result.get("search_type", "semantic"),
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
        source_filters: list[str] | None = None,
        use_hybrid: bool = True,
    ) -> dict:
        """
        Ask a question about the uploaded documents.
        Results are cached to avoid redundant LLM calls.
        """
        search_query = self._build_search_query(question, session_id)

        # Build cache key from all parameters that affect the result
        filters_key = ",".join(sorted(source_filters)) if source_filters else "all"
        history_key = str(len(self._get_history(session_id)))
        cache_key = ResponseCache._make_key(
            search_query, k, filters_key, use_hybrid, history_key
        )

        # Check answer cache (skip for conversations with history — answers depend on context)
        if not self._get_history(session_id):
            cached_answer = self.answer_cache.get(cache_key)
            if cached_answer:
                cached_answer["cached"] = True
                return cached_answer

        # Search (with search cache)
        search_cache_key = ResponseCache._make_key(search_query, k, filters_key, use_hybrid)
        cached_search = self.search_cache.get(search_cache_key)

        if cached_search:
            results = cached_search
        else:
            if use_hybrid:
                results = self.hybrid_search.search(
                    query=search_query,
                    k=k,
                    source_filters=source_filters,
                )
            else:
                results = self.vector_store.search(
                    query=search_query,
                    k=k,
                    source_filters=source_filters,
                )
            # Cache search results
            if results:
                self.search_cache.set(search_cache_key, results)

        if not results:
            msg = "No documents have been uploaded yet." if not source_filters else \
                  "No relevant content found in the selected documents."
            return {
                "answer": msg + " Please upload a legal document or adjust your filters.",
                "sources": [],
                "query": question,
                "num_sources": 0,
                "session_id": session_id,
                "source_filters": source_filters,
                "cached": False,
            }

        context = self._format_context(results)
        history_text = self._format_history(session_id)
        history_section = history_text if history_text else "No previous conversation."

        user_message = QUERY_TEMPLATE.format(
            context=context,
            history_section=history_section,
            question=question,
        )

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

        self._add_to_memory(session_id, "user", question)
        self._add_to_memory(session_id, "assistant", response.content)

        sources = self._extract_sources(results)

        result = {
            "answer": response.content,
            "sources": sources,
            "query": question,
            "num_sources": len(sources),
            "session_id": session_id,
            "source_filters": source_filters,
            "cached": False,
        }

        # Cache the answer (only for first-time questions without history)
        if not self._get_history(session_id) or len(self._get_history(session_id)) <= 2:
            self.answer_cache.set(cache_key, result)

        return result

    def invalidate_caches(self):
        """Clear all caches. Call after uploading/deleting documents."""
        self.search_cache.clear()
        self.answer_cache.clear()

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "search_cache": self.search_cache.stats(),
            "answer_cache": self.answer_cache.stats(),
        }

    def clear_memory(self, session_id: str = "default"):
        if session_id in self._memory:
            del self._memory[session_id]

    def get_memory(self, session_id: str = "default") -> list[dict]:
        return self._get_history(session_id)
