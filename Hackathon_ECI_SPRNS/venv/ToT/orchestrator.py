"""
Module: src/tot/orchestrator.py
Purpose: Tree-of-Thought (ToT) orchestration for LLM-driven KG fragment generation.

Improvements:
 - Uses central LLMClient abstraction (imported externally by callers).
 - Injects perception schema_context into prompts.
 - Stores LLM meta in StepResult and aggregates some meta at branch level.
 - Optional per-step canonicalization flag available.
 - Externalizable prompt templates (still defined here for convenience).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
import math
import uuid
import time
import logging

# canonicalization helper - may be patched to real module path in your repo
from Hackathon_ECI_SPRNS.venv.Schema.canonicalizer import canonicalize_nodes_relations
from Hackathon_ECI_SPRNS.venv.ToT.perceptions import get_perception  # schema injection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------
# Data containers
# ---------------------------

@dataclass
class StepResult:
    step_index: int
    prompt: str
    completion_text: str
    parsed_json: Optional[Dict[str, Any]] = None
    step_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    llm_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BranchResult:
    branch_id: str
    chunk_id: str
    perception: Optional[str]
    steps: List[StepResult] = field(default_factory=list)
    final_graph: Optional[Dict[str, Any]] = None
    loglik: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)


# ---------------------------
# Utilities
# ---------------------------

def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust-ish JSON extraction from LLM text.
    If parsing fails, returns None; callers should handle.
    """
    if not text or not text.strip():
        return None
    try:
        s = text.strip()
        if s[0] in ("{", "["):
            return json.loads(s)
        # find the first JSON block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            block = text[start:end + 1]
            return json.loads(block)
    except Exception as exc:
        logger.debug("safe_parse_json failed: %s", exc)
        return None
    return None


def aggregate_step_to_graph(current_graph: Dict[str, Any], step_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge nodes/relations from step_json into current_graph (union).
    Uniqueness heuristic: (label, span) for nodes, (from,to,type) for relations.
    """
    if current_graph is None:
        current_graph = {"nodes": [], "relations": []}
    existing_nodes = {(n.get("label"), n.get("span")) for n in current_graph.get("nodes", [])}
    for n in (step_json.get("nodes", []) if step_json else []):
        key = (n.get("label"), n.get("span"))
        if key not in existing_nodes:
            current_graph["nodes"].append(n)
            existing_nodes.add(key)

    existing_rels = {(r.get("from"), r.get("to"), r.get("type")) for r in current_graph.get("relations", [])}
    for r in (step_json.get("relations", []) if step_json else []):
        key = (r.get("from"), r.get("to"), r.get("type"))
        if key not in existing_rels:
            current_graph["relations"].append(r)
            existing_rels.add(key)
    return current_graph


def compute_step_score(parsed_json: Optional[Dict[str, Any]]) -> float:
    """
    Compute a proxy log-score for a step from confidences in parsed JSON.
    """
    if not parsed_json:
        return -5.0
    total, count = 0.0, 0
    eps = 1e-6
    for n in parsed_json.get("nodes", []):
        c = n.get("confidence", 0.5)
        c = max(min(float(c), 0.999999), eps)
        total += math.log(c)
        count += 1
    for r in parsed_json.get("relations", []):
        c = r.get("confidence", 0.5)
        c = max(min(float(c), 0.999999), eps)
        total += math.log(c)
        count += 1
    return float(total) if count > 0 else -2.0


# ---------------------------
# Prompt templates (configurable)
# ---------------------------

ROOT_PROMPT_TEMPLATE = """You are a Knowledge-Graph extractor.
Perception: {perception}
Schema guidance:
{schema_context}

Chunk text:
\"\"\"{chunk_text}\"\"\"

TASK:
Return JSON with arrays "nodes" and "relations".
Each node: {{ "label": "...", "span": "...", "confidence": 0-1, "props": {{}} }}
Each relation: {{ "from": "...", "to": "...", "type": "...", "confidence": 0-1 }}
If no new additions, return {{ "nodes": [], "relations": [] }}.
"""

STEP_PROMPT_TEMPLATE = """Continue KG extraction for Perception: {perception}
Schema guidance:
{schema_context}

Chunk text:
\"\"\"{chunk_text}\"\"\"

Partial graph so far (JSON): {partial_graph}

TASK:
Propose up to {max_add} new nodes/relations strictly in JSON format.
Each node/relation must include a confidence (0-1) and an example span.
If no new additions, return {{ "nodes": [], "relations": [] }}.
"""


# ---------------------------
# ToT Orchestrator
# ---------------------------

class ToTOrchestrator:
    def __init__(self, llm_client, max_steps: int = 5, max_add_per_step: int = 4):
        """
        llm_client: instance implementing 'complete(prompt)->(text,meta)' and optionally 'embed'
        """
        self.llm = llm_client
        self.max_steps = max_steps
        self.max_add_per_step = max_add_per_step

    def generate_branches(
            self,
            chunk: Dict[str, Any],
            perception: Optional[str] = None,
            max_branches: int = 3,
            temperature: float = 0.0,
            canonicalize_steps: bool = False,
    ) -> List[BranchResult]:
        """
        For a chunk, spawn up to max_branches ToT branches.
        canonicalize_steps: if True, canonicalizes each step's parsed JSON before merging (more stable downstream).
        """
        branches: List[BranchResult] = []

        schema_ctx = ""
        if perception:
            try:
                schema_ctx = get_perception(perception).to_prompt_context()
            except Exception:
                schema_ctx = ""

        for b in range(max_branches):
            branch_id = f"{chunk['chunk_id']}_b{b}_{uuid.uuid4().hex[:8]}"
            logger.info("Starting branch %s for chunk %s (perception=%s)", branch_id, chunk["chunk_id"], perception)
            br = BranchResult(
                branch_id=branch_id,
                chunk_id=chunk["chunk_id"],
                perception=perception,
                provenance={
                    "chunk_meta": {k: chunk.get(k) for k in ("doc_id", "entity_id", "start_char", "end_char")},
                    "branch_seed": b,
                    "schema_focus": schema_ctx,
                },
            )

            current_graph, cumulative_score = {"nodes": [], "relations": []}, 0.0
            aggregated_llm_meta: List[Dict[str, Any]] = []

            for step_idx in range(self.max_steps):
                prompt = ROOT_PROMPT_TEMPLATE.format(perception=(perception or "general"),
                                                     schema_context=schema_ctx,
                                                     chunk_text=chunk["raw_text"]) if step_idx == 0 else \
                    STEP_PROMPT_TEMPLATE.format(perception=(perception or "general"),
                                                schema_context=schema_ctx,
                                                chunk_text=chunk["raw_text"],
                                                partial_graph=json.dumps(current_graph),
                                                max_add=self.max_add_per_step)
                completion_text, meta = self.llm.complete(prompt, temperature=temperature)
                parsed = safe_parse_json(completion_text)
                step_score = compute_step_score(parsed)
                step = StepResult(
                    step_index=step_idx,
                    prompt=prompt,
                    completion_text=completion_text,
                    parsed_json=parsed,
                    step_score=step_score,
                    llm_meta=meta or {},
                )
                br.steps.append(step)
                aggregated_llm_meta.append(meta or {})

                if parsed:
                    step_to_merge = parsed
                    if canonicalize_steps:
                        try:
                            step_to_merge = canonicalize_nodes_relations(parsed, perception=perception)
                        except Exception as e:
                            logger.debug("Step canonicalization failed: %s", e)
                            step_to_merge = parsed
                    current_graph = aggregate_step_to_graph(current_graph, step_to_merge)

                cumulative_score += step_score

                # termination heuristics
                if parsed is not None and not parsed.get("nodes") and not parsed.get("relations"):
                    logger.info("Branch %s: early stop at step %d (no additions)", branch_id, step_idx)
                    break

            # canonicalize branch final graph if requested at branch level
            # (caller may also choose canonicalize_steps to do earlier)
            try:
                final_graph = canonicalize_nodes_relations(current_graph, perception=perception)
            except Exception:
                final_graph = current_graph

            br.final_graph = final_graph
            br.loglik = cumulative_score
            # aggregate llm meta for provenance (store small subset)
            br.provenance.update({
                "llm_meta_summary": {
                    "num_steps": len(br.steps),
                    "llm_sources": list({m.get("source") for m in aggregated_llm_meta if isinstance(m, dict) and m.get("source")}),
                },
            })
            branches.append(br)

        branches.sort(key=lambda x: x.loglik, reverse=True)
        return branches
