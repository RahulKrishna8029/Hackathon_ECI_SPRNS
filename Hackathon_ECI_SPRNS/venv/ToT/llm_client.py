"""
Module: src/tot/llm_client.py
Purpose: Provide LLMClient abstraction + implementations (OpenAI, Mock).
"""

from typing import Tuple, Dict, Any
import os
import openai


class LLMClient:
    """Abstract base client for LLM providers."""

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> Tuple[str, Dict[str, Any]]:
        """
        :param prompt: Prompt string to send to LLM.
        :param temperature: Sampling temperature.
        :param max_tokens: Max tokens to generate.
        :return: (completion_text, meta_dict)
        """
        raise NotImplementedError("Implement in subclass.")


class OpenAIClient(LLMClient):
    """OpenAI GPT client (Chat Completions API)."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512):
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise Knowledge Graph extractor."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp["choices"][0]["message"]["content"]
        meta = {
            "usage": resp.get("usage", {}),
            "finish_reason": resp["choices"][0]["finish_reason"],
        }
        return text, meta


class MockLLMClient(LLMClient):
    """Mock client for local dev/testing without API cost."""

    def __init__(self, canned_responses=None):
        self.canned = canned_responses or []

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512):
        if self.canned:
            return self.canned.pop(0), {}
        return '{"nodes":[{"label":"Name","span":"John Doe","confidence":0.95}],"relations":[]}', {}