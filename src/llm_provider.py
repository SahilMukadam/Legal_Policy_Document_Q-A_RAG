"""
LLM Provider Factory.

Returns the configured LLM instance based on LLM_PROVIDER in .env.
Swap providers by changing one line in .env — no code changes needed.

Usage:
    from src.llm_provider import get_llm
    llm = get_llm()
    response = llm.invoke("What is a contract?")
"""

from configs.settings import settings


def get_llm():
    """
    Returns a LangChain LLM instance based on the configured provider.

    Supported providers:
        - "gemini"    → Google Gemini 1.5 Flash (free tier)
        - "anthropic" → Anthropic Claude (paid)

    To switch providers:
        1. Update LLM_PROVIDER in your .env file
        2. Ensure the corresponding API key is set
        3. That's it — no code changes needed
    """
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set in .env. "
                "Get your free key at: https://aistudio.google.com/apikey"
            )

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            max_output_tokens=settings.max_tokens,
            temperature=0.1,  # Low temp for factual Q&A
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set in .env. "
                "Get your key at: https://console.anthropic.com/"
            )

        return ChatAnthropic(
            model=settings.claude_model,
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=settings.max_tokens,
            temperature=0.1,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Supported: 'gemini', 'anthropic'"
        )
