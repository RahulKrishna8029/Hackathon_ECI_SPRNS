"""
Module: src/kg/graph_constructor.py
Purpose: Construct canonical GraphFragments from ToT outputs.
"""

import uuid
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

from Hackathon_ECI_SPRNS.venv.Schema.canonicalizer import canonicalize_nodes_relations, _embed_label

logger = logging.getLogger(__name__)


# ---------------------------
# Data container
# ---------------------------

@dataclass
class GraphFragment:
    fragment_id: str
    entity_id: str
    nodes: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    phi_raw: np.ndarray
    provenance: Dict[str, Any]


# ---------------------------
# Helpers
# ---------------------------

def _aggregate_schema_embeddings(graph: Dict[str, Any]) -> np.ndarray:
    """
    Aggregate schema embeddings of all nodes + relations in the graph.
    """
    vecs = []
    for n in graph.get("nodes", []):
        if "label" in n:
            vecs.append(_embed_label(n["label"]))
    for r in graph.get("relations", []):
        if "type" in r:
            vecs.append(_embed_label(r["type"]))

    if not vecs:
        return np.zeros(128)

    mat = np.vstack(vecs)
    return np.mean(mat, axis=0)


def _build_adjacency(graph: Dict[str, Any], uid_dim: int = 128) -> np.ndarray:
    """
    Simplified adjacency encoding: sum of outer products of node embeddings.
    Uses source/dest labels, not raw UIDs.
    """
    A = np.zeros((uid_dim, uid_dim))
    nodes = {n.get("uid"): n.get("label") for n in graph.get("nodes", []) if "uid" in n}
    for r in graph.get("relations", []):
        src_label = nodes.get(r.get("from"), str(r.get("from", "")))
        dst_label = nodes.get(r.get("to"), str(r.get("to", "")))
        src = _embed_label(src_label)
        dst = _embed_label(dst_label)
        A += np.outer(src, dst)
    return A


# ---------------------------
# Constructor
# ---------------------------

def construct_fragment(branch_result, canonicalize: bool = True) -> GraphFragment:
    """
    Convert BranchResult.final_graph into a canonical GraphFragment.
    """
    graph = branch_result.final_graph or {"nodes": [], "relations": []}

    # Canonicalize if requested
    if canonicalize:
        graph = canonicalize_nodes_relations(graph, perception=branch_result.perception)

    # Compute φ_raw = schema aggregate + adjacency signature
    phi_schema = _aggregate_schema_embeddings(graph)
    A = _build_adjacency(graph)
    phi_raw = np.concatenate([phi_schema, A.mean(axis=0)])  # simple flattening

    # Defensive entity_id retrieval
    entity_id = (
            branch_result.provenance.get("chunk_meta", {}).get("entity_id")
            or getattr(branch_result, "entity_id", None)
            or "unknown"
    )

    # Warn if empty graph
    if not graph.get("nodes") and not graph.get("relations"):
        logger.warning("Empty graph in fragment %s (perception=%s)",
                       branch_result.branch_id, branch_result.perception)

    return GraphFragment(
        fragment_id=str(uuid.uuid4()),
        entity_id=entity_id,
        nodes=graph.get("nodes", []),
        relations=graph.get("relations", []),
        phi_raw=phi_raw,
        provenance={
            "branch_id": branch_result.branch_id,
            "perception": branch_result.perception,
            "loglik": branch_result.loglik,
            "chunk_meta": branch_result.provenance.get("chunk_meta", {}),
        },
    )
