"""Quick test to verify Gemini API connection works."""

import sys
sys.path.append(".")

from src.llm_provider import get_llm


def test_gemini():
    llm = get_llm()
    response = llm.invoke("Say 'Hello, Legal Q&A!' and nothing else.")
    print(f"Provider: Gemini")
    print(f"Response: {response.content}")
    print("✅ Gemini connection working!")


if __name__ == "__main__":
    test_gemini()