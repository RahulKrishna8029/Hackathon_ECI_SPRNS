"""
Module: src/encode/encoder.py
Purpose: Encode GraphFragments into embeddings using LLM embeddings.
"""

import numpy as np
from typing import List
import logging

from Hackathon_ECI_SPRNS.venv.KG.graph_constructor import GraphFragment

logger = logging.getLogger(__name__)


# ---------------------------
# Utilities
# ---------------------------

def fragment_to_text(fragment: GraphFragment) -> str:
    """
    Serialize a GraphFragment into plain text for LLM embedding.
    Defensive: tolerate missing keys.
    """
    node_strs = [f"{n.get('uid','?')}:{n.get('label','?')}" for n in fragment.nodes]
    rel_strs = [
        f"{r.get('from','?')} -[{r.get('type','?')}]-> {r.get('to','?')}"
        for r in fragment.relations
    ]
    text = f"Nodes: {', '.join(node_strs)}\nRelations: {', '.join(rel_strs)}"
    return text.strip()


# ---------------------------
# Encoder
# ---------------------------

# src/encode/encoder.py

class GraphEncoderLLM:
    def __init__(self, llm_client, model: str = "text-embedding-3-large", batch_size: int = 16):
        """
        :param llm_client: must implement `.embed(text, model)` returning vector
        :param model: embedding model name (default OpenAI text-embedding-3-large)
        :param batch_size: batch size for API calls (if supported by llm_client)
        """
        self.llm_client = llm_client
        self.model = model
        self.batch_size = batch_size

    def embed_fragment(self, fragment: GraphFragment, persist: bool = True) -> np.ndarray:
        """
        Compute embedding for a single GraphFragment and (optionally) persist it.
        """
        text = fragment_to_text(fragment)
        try:
            vector = self.llm_client.embed(text, self.model)
        except Exception as e:
            logger.error("Embedding failed for fragment %s: %s", fragment.fragment_id, e)
            vector = np.zeros(128).tolist()  # fallback

        vec = np.array(vector, dtype=float)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if persist:
            fragment.phi_emb = vec  # ✅ attach directly
            fragment.provenance["embedding_model"] = self.model
            fragment.provenance["embedding_dim"] = len(vec)

        return vec

    def embed_fragments(self, fragments: List[GraphFragment], persist: bool = True) -> List[np.ndarray]:
        """
        Compute embeddings for multiple fragments.
        Batch texts when possible (if llm_client supports batching).
        """
        texts = [fragment_to_text(f) for f in fragments]
        vectors: List[np.ndarray] = []

        # If llm_client supports batch embedding
        if hasattr(self.llm_client, "embed_batch"):
            logger.info("Using batch embedding (size=%d)", self.batch_size)
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                try:
                    batch_vecs = self.llm_client.embed_batch(batch_texts, self.model)
                except Exception as e:
                    logger.error("Batch embedding failed: %s", e)
                    batch_vecs = [np.zeros(128).tolist() for _ in batch_texts]

                for f, vec in zip(fragments[i:i+self.batch_size], batch_vecs):
                    arr = np.array(vec, dtype=float)
                    arr /= np.linalg.norm(arr) if np.linalg.norm(arr) > 0 else 1
                    if persist:
                        f.phi_emb = arr
                        f.provenance["embedding_model"] = self.model
                        f.provenance["embedding_dim"] = len(arr)
                    vectors.append(arr)
        else:
            # fallback: sequential
            for f in fragments:
                vectors.append(self.embed_fragment(f, persist=persist))

        return vectors
