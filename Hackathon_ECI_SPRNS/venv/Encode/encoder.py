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

class GraphEncoderLLM:
    def __init__(self, llm_client, model: str = "text-embedding-3-large", expected_dim: int = 3072):
        """
        :param llm_client: must implement `.embed(text, model)` returning vector
        :param model: embedding model name
        :param expected_dim: expected embedding dimensionality (default 3072 for OpenAI large model)
        """
        self.llm_client = llm_client
        self.model = model
        self.expected_dim = expected_dim

    def embed_fragment(self, fragment: GraphFragment, persist: bool = True) -> np.ndarray:
        """
        Compute embedding for a single GraphFragment and (optionally) persist it.
        """
        text = fragment_to_text(fragment)
        try:
            vector = self.llm_client.embed(text, self.model)
        except Exception as e:
            logger.error("Embedding failed for fragment %s: %s", fragment.fragment_id, e)
            vector = np.zeros(self.expected_dim).tolist()  # fallback
            failure = True
        else:
            failure = False

        vec = np.array(vector, dtype=float)

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            logger.warning("Zero vector produced for fragment %s", fragment.fragment_id)

        if persist:
            # Attach embedding into fragment (typed)
            fragment.provenance["embedding_model"] = self.model
            fragment.provenance["embedding_dim"] = len(vec)
            fragment.provenance["embedding_failed"] = failure
            # Add phi_emb explicitly
            if not hasattr(fragment, "phi_emb"):
                setattr(fragment, "phi_emb", vec)
            else:
                fragment.phi_emb = vec

        return vec

    def embed_fragments(self, fragments: List[GraphFragment], persist: bool = True) -> List[np.ndarray]:
        """
        Compute embeddings for multiple fragments.
        NOTE: Uses naive loop; for OpenAI you could batch to save cost.
        """
        results = []
        for f in fragments:
            vec = self.embed_fragment(f, persist=persist)
            if vec.shape[0] != self.expected_dim:
                logger.warning("Fragment %s embedding dim %d != expected %d",
                               f.fragment_id, vec.shape[0], self.expected_dim)
            results.append(vec)
        return results
