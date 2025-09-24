"""
Module: src/schema/canonicalizer.py
Purpose: Discover schema entities (nodes/relations), merge synonyms,
and assign canonical UIDs with persistence.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import uuid
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import datetime

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

# Optional external embedder (e.g. LLMClient). Can be injected at runtime.
EMBEDDER = None


# ---------------------------
# Utilities
# ---------------------------

def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _save_registry(path: str, registry: Dict[str, Any]):
    _ensure_data_dir()
    # convert numpy arrays to lists for JSON
    prepared = {}
    for uid, meta in registry.items():
        m = meta.copy()
        if isinstance(m.get("embedding"), np.ndarray):
            m["embedding"] = m["embedding"].tolist()
        prepared[uid] = m
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prepared, f, indent=2)


def _load_registry(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # ensure embeddings are numpy arrays
    for uid, meta in data.items():
        if isinstance(meta.get("embedding"), list):
            meta["embedding"] = np.array(meta["embedding"], dtype=float)
    return data


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

def _embed_label(label: str, model: str = "text-embedding-3-large") -> np.ndarray:
    """
    Embedding backend for schema labels.
    - If EMBEDDER is set: call EMBEDDER.embed([label], model).
    - Else: fallback to deterministic hash vector.
    """
    if not label:
        return np.zeros(128)

    global EMBEDDER
    if EMBEDDER is not None:
        try:
            vecs = EMBEDDER.embed([label], model=model)
            return np.array(vecs[0], dtype=float)
        except Exception as e:
            logger.warning("LLM embedder failed for '%s': %s", label, e)

    # fallback: deterministic random
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
        vec = meta.get("embedding")
        if isinstance(vec, list):
            vec = np.array(vec, dtype=float)
        if isinstance(vec, np.ndarray):
            uids.append(uid)
            vecs.append(vec)
    if not vecs:
        return None, 0.0

    sims = cosine_similarity([emb], vecs)[0]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= threshold:
        return uids[best_idx], float(sims[best_idx])
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
    """
    if not label:
        raise ValueError("Cannot canonicalize empty label")

    emb = _embed_label(label)
    registry = RELATION_REGISTRY if is_relation else NODE_REGISTRY
    match_uid, sim = _find_match(emb, registry, threshold)

    if match_uid:
        # update aliases
        aliases = set(registry[match_uid].get("aliases", []))
        aliases.add(label)
        registry[match_uid]["aliases"] = list(sorted(aliases))
        return match_uid

    # Create new UID
    prefix = "rel" if is_relation else "node"
    uid = f"schema::{prefix}::{label.lower()}::{uuid.uuid4().hex[:6]}"
    registry[uid] = {
        "label": label,
        "embedding": emb,
        "perception": perception,
        "aliases": [label],
        "created_at": datetime.datetime.utcnow().isoformat(),
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
    node_map: Dict[str, str] = {}

    # Canonicalize nodes
    for n in graph_json.get("nodes", []):
        label = n.get("label")
        if not label:
            logger.warning("Skipping node with missing label: %s", n)
            continue
        try:
            uid = canonicalize_label(label, perception=perception, threshold=threshold, is_relation=False)
        except ValueError:
            continue
        key = n.get("span") or label
        node_map[key] = uid
        new_graph["nodes"].append({
            "uid": uid,
            "label": label,
            "span": n.get("span"),
            "confidence": n.get("confidence", 0.5),
            "props": n.get("props", {}),
            "perception": perception,
        })

    # Canonicalize relations
    for r in graph_json.get("relations", []):
        rtype = r.get("type")
        if not rtype:
            logger.warning("Skipping relation with missing type: %s", r)
            continue
        try:
            rel_uid = canonicalize_label(rtype, perception=perception, threshold=threshold, is_relation=True)
        except ValueError:
            continue

        from_uid = node_map.get(r.get("from")) or r.get("from")
        to_uid = node_map.get(r.get("to")) or r.get("to")

        new_graph["relations"].append({
            "uid": rel_uid,
            "type": rtype,
            "from": from_uid,
            "to": to_uid,
            "confidence": r.get("confidence", 0.5),
            "perception": perception,
        })

    return new_graph


# ---------------------------
# Init
# ---------------------------

load_registries()
