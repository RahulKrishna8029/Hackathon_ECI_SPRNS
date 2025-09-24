"""
Module: src/tot/llm_client.py
Purpose: Provide LLMClient abstraction + implementations (OpenAI, Mock).
"""

from typing import Tuple, Dict, Any, List
import os
import openai


# ---------------------------
# Base abstraction
# ---------------------------

class LLMClient:
    """Abstract base client for LLM providers."""

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> Tuple[str, Dict[str, Any]]:
        """LLM completion (chat/text generation)."""
        raise NotImplementedError("Implement in subclass.")

    def embed(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        """LLM embeddings (vector representation of text)."""
        raise NotImplementedError("Implement in subclass.")


# ---------------------------
# OpenAI client
# ---------------------------

class OpenAIClient(LLMClient):
    """OpenAI GPT + Embeddings client."""

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

    def embed(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        resp = openai.Embedding.create(
            model=model,
            input=text,
        )
        return resp["data"][0]["embedding"]

# ---------------------------
# Mock client
# ---------------------------

class MockLLMClient(LLMClient):
    """Mock client for local dev/testing without API cost."""

    def __init__(self, canned_responses=None, mock_embedding_dim: int = 128):
        self.canned = canned_responses or []
        self.dim = mock_embedding_dim

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512):
        if self.canned:
            return self.canned.pop(0), {}
        return '{"nodes":[{"label":"Name","span":"John Doe","confidence":0.95}],"relations":[]}', {}

    def embed(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        # simple deterministic hash → embedding for repeatability
        import numpy as np
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(self.dim).tolist()
