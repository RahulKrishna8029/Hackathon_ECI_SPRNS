"""
Module: src/schema/canonicalizer.py
Purpose: Discover schema entities (nodes/relations), merge synonyms,
and assign canonical UIDs with persistence.
"""

from typing import List, Dict, Any, Tuple, Optional
import uuid
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Persistent registries
# ---------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
NODE_REGISTRY_PATH = os.path.join(DATA_DIR, "node_registry.json")
REL_REGISTRY_PATH = os.path.join(DATA_DIR, "relation_registry.json")

NODE_REGISTRY: Dict[str, Dict[str, Any]] = {}
RELATION_REGISTRY: Dict[str, Dict[str, Any]] = {}


# ---------------------------
# Utilities
# ---------------------------

def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _save_registry(path: str, registry: Dict[str, Any]):
    _ensure_data_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def _load_registry(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_registries():
    """Load registries from disk into memory."""
    global NODE_REGISTRY, RELATION_REGISTRY
    NODE_REGISTRY = _load_registry(NODE_REGISTRY_PATH)
    RELATION_REGISTRY = _load_registry(REL_REGISTRY_PATH)


def save_registries():
    """Persist registries to disk."""
    _save_registry(NODE_REGISTRY_PATH, NODE_REGISTRY)
    _save_registry(REL_REGISTRY_PATH, RELATION_REGISTRY)


# ---------------------------
# Embedding backend (pluggable)
# ---------------------------

def _embed_label(label: str) -> np.ndarray:
    """
    Embedding backend for schema labels.
    Currently: deterministic random hash → vector (128D).
    Replace with real model (e.g., sentence-transformers, OpenAI embeddings).
    """
    if not label:
        return np.zeros(128)
    np.random.seed(abs(hash(label)) % (2**32))
    return np.random.rand(128)


def _find_match(
        emb: np.ndarray,
        registry: Dict[str, Dict[str, Any]],
        threshold: float = 0.85,
) -> Tuple[Optional[str], float]:
    """
    Try to find the best matching UID in registry.
    """
    if not registry:
        return None, 0.0
    uids, vecs = [], []
    for uid, meta in registry.items():
        if "embedding" in meta:
            uids.append(uid)
            vecs.append(meta["embedding"])
    if not vecs:
        return None, 0.0
    sims = cosine_similarity([emb], vecs)[0]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= threshold:
        return uids[best_idx], sims[best_idx]
    return None, 0.0


# ---------------------------
# Canonicalization functions
# ---------------------------

def canonicalize_label(
        label: str,
        perception: Optional[str] = None,
        threshold: float = 0.85,
        is_relation: bool = False,
) -> str:
    """
    Map a raw label to a canonical UID (creating if needed).
    :param label: raw label string
    :param perception: optional perception category (identity, txn, etc.)
    :param threshold: cosine similarity threshold
    :param is_relation: if True → relation registry; else → node registry
    """
    if not label:
        raise ValueError("Cannot canonicalize empty label")

    emb = _embed_label(label)
    registry = RELATION_REGISTRY if is_relation else NODE_REGISTRY
    match_uid, sim = _find_match(emb, registry, threshold)

    if match_uid:
        return match_uid

    # Create new UID
    prefix = "rel" if is_relation else "node"
    uid = f"schema::{prefix}::{label.lower()}::{uuid.uuid4().hex[:6]}"
    registry[uid] = {
        "label": label,
        "embedding": emb.tolist(),  # store as list for JSON
        "perception": perception,
        "aliases": [label],
    }

    save_registries()
    return uid


def canonicalize_nodes_relations(
        graph_json: Dict[str, Any],
        perception: Optional[str] = None,
        threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Given a graph fragment (nodes/relations from ToT), map all labels to canonical UIDs.
    Endpoints in relations are remapped to canonical node UIDs.
    """
    new_graph = {"nodes": [], "relations": []}
    node_map = {}  # raw span/label → canonical UID

    # Canonicalize nodes
    for n in graph_json.get("nodes", []):
        label = n.get("label")
        try:
            uid = canonicalize_label(label, perception=perception, threshold=threshold, is_relation=False)
        except ValueError:
            continue
        node_map[(label, n.get("span"))] = uid
        new_graph["nodes"].append({
            "uid": uid,
            "label": label,
            "span": n.get("span"),
            "confidence": n.get("confidence", 0.5),
            "props": n.get("props", {}),
        })

    # Canonicalize relations
    for r in graph_json.get("relations", []):
        rtype = r.get("type")
        try:
            rel_uid = canonicalize_label(rtype, perception=perception, threshold=threshold, is_relation=True)
        except ValueError:
            continue

        # map endpoints to canonical UIDs if available
        from_uid = node_map.get((r.get("from"), None), r.get("from"))
        to_uid = node_map.get((r.get("to"), None), r.get("to"))

        new_graph["relations"].append({
            "uid": rel_uid,
            "type": rtype,
            "from": from_uid,
            "to": to_uid,
            "confidence": r.get("confidence", 0.5),
        })

    return new_graph


# ---------------------------
# Init
# ---------------------------

# Load registries on import
load_registries()