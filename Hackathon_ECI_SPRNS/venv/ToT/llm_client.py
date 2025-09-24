"""
Module: src/tot/llm_client.py
Purpose: Provide LLMClient abstraction + implementations (OpenAI, Mock).
Notes:
 - Includes simple retry/backoff for API robustness.
 - Embedding method supports batching.
"""

from typing import Tuple, Dict, Any, List, Optional
import os
import time
import math
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
    openai = None  # type: ignore

# ---------------------------
# Base abstraction
# ---------------------------

class LLMClient:
    """Abstract base client for LLM providers."""

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> Tuple[str, Dict[str, Any]]:
        """LLM completion (chat/text generation)."""
        raise NotImplementedError("Implement in subclass.")

    def embed(self, texts: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
        """LLM embeddings (vector representation of text). Accepts list of texts for batching."""
        raise NotImplementedError("Implement in subclass.")


# ---------------------------
# OpenAI client
# ---------------------------

class OpenAIClient(LLMClient):
    """OpenAI GPT + Embeddings client with simple retry/backoff."""

    def __init__(self, model: str = "gpt-4o-mini", embed_model: str = "text-embedding-3-large", max_retries: int = 3):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai package not available. Install 'openai' or use MockLLMClient for testing.")
        self.model = model
        self.embed_model = embed_model
        self.max_retries = max_retries
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def _retry(self, fn, *args, **kwargs):
        backoff = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.warning("OpenAI call failed (attempt %d/%d): %s", attempt, self.max_retries, e)
                if attempt == self.max_retries:
                    raise
                time.sleep(backoff)
                backoff *= 2.0 + 0.5 * attempt

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> Tuple[str, Dict[str, Any]]:
        """
        Chat completion using OpenAI Chat API. Returns (text, meta).
        Note: works with both older and newer openai SDK shapes; tries common fields.
        """
        def _call():
            # prefer chat.completions if available on client; fallback to legacy create
            if hasattr(openai, "ChatCompletion"):
                resp = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise Knowledge Graph extractor."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                choice = resp["choices"][0]
                text = choice.get("message", {}).get("content", choice.get("text", ""))
                meta = {"usage": resp.get("usage", {}), "finish_reason": choice.get("finish_reason")}
                return text, meta
            else:
                # fallback: simple Completion endpoint
                resp = openai.Completion.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = resp["choices"][0].get("text", "")
                meta = {"usage": resp.get("usage", {}), "finish_reason": resp["choices"][0].get("finish_reason")}
                return text, meta

        return self._retry(_call)

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Batch embedding via OpenAI embeddings API.
        Returns list of vectors aligned with `texts`.
        """
        model = model or self.embed_model

        def _call():
            # openai.Embedding.create accepts list inputs
            resp = openai.embeddings.create(model=model, input=texts)
            return [item["embedding"] for item in resp["data"]]

        return self._retry(_call)


# ---------------------------
# Mock client
# ---------------------------

class MockLLMClient(LLMClient):
    """Mock client for local dev/testing without API cost. Keeps deterministic embeddings."""

    def __init__(self, canned_responses: Optional[List[str]] = None, mock_embedding_dim: int = 128):
        self.canned = canned_responses or []
        self.dim = mock_embedding_dim

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> Tuple[str, Dict[str, Any]]:
        if self.canned:
            return self.canned.pop(0), {"source": "mock"}
        # minimal deterministic response
        sample = '{"nodes":[{"label":"Name","span":"John Doe","confidence":0.95}],"relations":[]}'
        return sample, {"source": "mock"}

    def embed(self, texts: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
        import numpy as np
        out = []
        for t in texts:
            np.random.seed(abs(hash(t)) % (2**32))
            out.append(np.random.rand(self.dim).tolist())
        return out
