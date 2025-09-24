"""
Module: src/kg/graph_constructor.py
Purpose: Construct canonical GraphFragments from ToT outputs.
"""

import uuid
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from Hackathon_ECI_SPRNS.venv.Schema import canonicalizer

logger = logging.getLogger(__name__)


# ---------------------------
# Data container
# ---------------------------

@dataclass
# src/kg/graph_constructor.py

@dataclass
class GraphFragment:
    fragment_id: str
    entity_id: str
    nodes: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    phi_raw: np.ndarray
    provenance: Dict[str, Any]
    phi_emb: Optional[np.ndarray] = None  # ✅ LLM embedding, set later by encoder



# ---------------------------
# Helpers
# ---------------------------

def _embed_any(label_or_uid: str) -> np.ndarray:
    """
    Wrapper around canonicalizer._embed_label, which may call LLM embedder if available.
    """
    return canonicalizer._embed_label(label_or_uid)


def _aggregate_schema_embeddings(graph: Dict[str, Any]) -> np.ndarray:
    """
    Aggregate schema embeddings of all nodes + relations in the graph.
    """
    vecs = []
    for n in graph.get("nodes", []):
        if "uid" in n:
            vecs.append(_embed_any(n["uid"]))
        elif "label" in n:
            vecs.append(_embed_any(n["label"]))
    for r in graph.get("relations", []):
        if "uid" in r:
            vecs.append(_embed_any(r["uid"]))
        elif "type" in r:
            vecs.append(_embed_any(r["type"]))

    if not vecs:
        return np.zeros(128)

    mat = np.vstack(vecs)
    mean_vec = np.mean(mat, axis=0)
    return mean_vec / (np.linalg.norm(mean_vec) + 1e-8)


def _build_adjacency_embedding(graph: Dict[str, Any], uid_dim: int = 128, max_dim: int = 256) -> np.ndarray:
    """
    Build adjacency signature as low-rank embedding.
    - Compute adjacency outer products in embedding space.
    - Compress with SVD to fixed dim (max_dim).
    """
    A = np.zeros((uid_dim, uid_dim))
    nodes = {n.get("uid", n.get("label")): n for n in graph.get("nodes", [])}
    for r in graph.get("relations", []):
        src_key = r.get("from")
        dst_key = r.get("to")
        src_label = nodes.get(src_key, {}).get("uid") or nodes.get(src_key, {}).get("label") or str(src_key)
        dst_label = nodes.get(dst_key, {}).get("uid") or nodes.get(dst_key, {}).get("label") or str(dst_key)
        src = _embed_any(src_label)
        dst = _embed_any(dst_label)
        A += np.outer(src, dst)

    if not np.any(A):
        return np.zeros(max_dim)

    # Low-rank projection via SVD
    try:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        vec = (U[:, 0] * S[0]) if S.size > 0 else np.zeros(uid_dim)
    except Exception as e:
        logger.warning("Adjacency SVD failed: %s", e)
        vec = A.mean(axis=0)

    # Compress or pad to max_dim
    if vec.shape[0] > max_dim:
        vec = vec[:max_dim]
    elif vec.shape[0] < max_dim:
        pad = np.zeros(max_dim)
        pad[: vec.shape[0]] = vec
        vec = pad

    return vec / (np.linalg.norm(vec) + 1e-8)


# ---------------------------
# Constructor
# ---------------------------

def construct_fragment(branch_result, canonicalize: bool = False) -> GraphFragment:
    """
    Convert BranchResult.final_graph into a canonical GraphFragment.
    """
    graph = branch_result.final_graph or {"nodes": [], "relations": []}

    # Canonicalize if requested
    if canonicalize:
        graph = canonicalizer.canonicalize_nodes_relations(graph, perception=branch_result.perception)
    else:
        if not any("uid" in n for n in graph.get("nodes", [])):
            logger.warning("Graph passed without canonicalization and no UIDs found (branch=%s)", branch_result.branch_id)

    # Compute φ_raw = schema aggregate + adjacency signature
    phi_schema = _aggregate_schema_embeddings(graph)
    phi_adj = _build_adjacency_embedding(graph)
    phi_raw = np.concatenate([phi_schema, phi_adj])

    # Defensive entity_id retrieval
    entity_id = (
            branch_result.provenance.get("chunk_meta", {}).get("entity_id")
            or branch_result.provenance.get("entity_id")
            or getattr(branch_result, "entity_id", None)
            or "unknown"
    )
    if entity_id == "unknown":
        logger.warning("Entity ID missing for fragment %s", branch_result.branch_id)

    # Provenance enrichment
    provenance = {
        "branch_id": branch_result.branch_id,
        "perception": branch_result.perception,
        "loglik": branch_result.loglik,
        "chunk_meta": branch_result.provenance.get("chunk_meta", {}),
        "processed_chunks": branch_result.provenance.get("processed_chunks", []),
        "schema_focus": branch_result.provenance.get("schema_focus"),
    }

    # Logging
    num_nodes, num_rels = len(graph.get("nodes", [])), len(graph.get("relations", []))
    if num_nodes == 0 and num_rels == 0:
        logger.warning("Empty graph in fragment %s (perception=%s)", branch_result.branch_id, branch_result.perception)
    else:
        logger.info("Constructed fragment %s with %d nodes and %d relations (perception=%s)",
                    branch_result.branch_id, num_nodes, num_rels, branch_result.perception)

    return GraphFragment(
        fragment_id=str(uuid.uuid4()),
        entity_id=entity_id,
        nodes=graph.get("nodes", []),
        relations=graph.get("relations", []),
        phi_raw=phi_raw,
        provenance=provenance,
    )
