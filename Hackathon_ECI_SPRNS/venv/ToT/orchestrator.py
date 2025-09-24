"""
Module: src/tot/orchestrator.py
Purpose: Tree-of-Thought (ToT) orchestration for LLM-driven KG fragment generation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
import math
import uuid
import time
import logging

from Hackathon_ECI_SPRNS.venv.Schema.canonicalizer import canonicalize_nodes_relations
from Hackathon_ECI_SPRNS.venv.ToT.perceptions import get_perception  # ✅ new import

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
    llm_meta: Dict[str, Any] = field(default_factory=dict)  # ✅ meta saved


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
# LLM client abstraction
# ---------------------------

class LLMClient:
    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError("Implement provider-specific complete()")


class MockLLMClient(LLMClient):
    def __init__(self, canned_responses: Optional[List[str]] = None):
        self.canned = canned_responses or []

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512):
        if self.canned:
            text = self.canned.pop(0)
            return text, {"source": "mock"}
        resp = {
            "nodes": [
                {"label": "Address", "span": "12 High St", "confidence": 0.92}
            ],
            "relations": []
        }
        return json.dumps(resp), {"source": "mock"}


# ---------------------------
# Utilities
# ---------------------------

def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text or not text.strip():
        return None
    try:
        text_strip = text.strip()
        if text_strip[0] in ["{", "["]:
            return json.loads(text_strip)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            return json.loads(candidate)
    except Exception as exc:
        logger.debug("JSON parse failed: %s -- exc=%s", text[:200], exc)
        return None
    return None


def aggregate_step_to_graph(current_graph: Dict[str, Any], step_json: Dict[str, Any]) -> Dict[str, Any]:
    if current_graph is None:
        current_graph = {"nodes": [], "relations": []}
    existing_nodes = {(n.get("label"), n.get("span")) for n in current_graph.get("nodes", [])}
    for n in step_json.get("nodes", []) if step_json else []:
        key = (n.get("label"), n.get("span"))
        if key not in existing_nodes:
            current_graph["nodes"].append(n)
            existing_nodes.add(key)
    existing_rels = {(r.get("from"), r.get("to"), r.get("type")) for r in current_graph.get("relations", [])}
    for r in step_json.get("relations", []) if step_json else []:
        key = (r.get("from"), r.get("to"), r.get("type"))
        if key not in existing_rels:
            current_graph["relations"].append(r)
            existing_rels.add(key)
    return current_graph


def compute_step_score(parsed_json: Optional[Dict[str, Any]]) -> float:
    if not parsed_json:
        return -5.0
    total, count, eps = 0.0, 0, 1e-6
    for n in parsed_json.get("nodes", []):
        c = n.get("confidence", None)
        if c is None:
            total += math.log(0.5 + eps)
        else:
            c = max(min(float(c), 0.999999), 1e-6)
            total += math.log(c)
        count += 1
    for r in parsed_json.get("relations", []):
        c = r.get("confidence", None)
        if c is None:
            total += math.log(0.5 + eps)
        else:
            c = max(min(float(c), 0.999999), 1e-6)
            total += math.log(c)
        count += 1
    if count == 0:
        return -2.0
    return float(total)


# ---------------------------
# Prompt templates
# ---------------------------

STEP_PROMPT_TEMPLATE = """You are continuing KG extraction for Perception: {perception}
Schema guidance:
{schema_context}

Chunk:
\"\"\"{chunk_text}\"\"\"

Partial graph so far (JSON): {partial_graph}

TASK:
Propose up to {max_add} augmentations (nodes/relations) strictly in JSON format as:
{{ "nodes":[...], "relations":[...] }}.
Each node/relation must include a confidence (0-1).
If no new additions, return {{ "nodes": [], "relations": [] }}.
"""


# ---------------------------
# ToT Orchestrator
# ---------------------------

class ToTOrchestrator:
    def __init__(self, llm_client: LLMClient, max_steps: int = 5, max_add_per_step: int = 4):
        self.llm = llm_client
        self.max_steps = max_steps
        self.max_add_per_step = max_add_per_step

    def generate_branches(
            self,
            chunk: Dict[str, Any],
            perception: Optional[str] = None,
            max_branches: int = 3,
            temperature: float = 0.0,
            canonicalize: bool = False,
    ) -> List[BranchResult]:
        branches: List[BranchResult] = []

        schema_ctx = ""
        if perception:
            try:
                schema_ctx = get_perception(perception).to_prompt_context()
            except Exception:
                schema_ctx = ""

        for b in range(max_branches):
            branch_id = f"{chunk['chunk_id']}_b{b}_{uuid.uuid4().hex[:8]}"
            logger.info("Starting branch %s for chunk %s", branch_id, chunk["chunk_id"])
            br = BranchResult(
                branch_id=branch_id,
                chunk_id=chunk["chunk_id"],
                perception=perception,
                provenance={
                    "chunk_meta": {k: chunk.get(k) for k in ("doc_id", "entity_id", "start_char", "end_char")},
                    "branch_seed": b,
                    "schema_focus": schema_ctx,  # ✅ provenance includes schema
                },
            )
            current_graph, cumulative_score = {"nodes": [], "relations": []}, 0.0
            for step_idx in range(self.max_steps):
                prompt = STEP_PROMPT_TEMPLATE.format(
                    perception=(perception or "general"),
                    schema_context=schema_ctx,
                    chunk_text=chunk["raw_text"],
                    partial_graph=json.dumps(current_graph),
                    max_add=self.max_add_per_step,
                )
                completion_text, meta = self.llm.complete(prompt, temperature=temperature)
                parsed = safe_parse_json(completion_text)
                step_score = compute_step_score(parsed)
                step = StepResult(
                    step_index=step_idx,
                    prompt=prompt,
                    completion_text=completion_text,
                    parsed_json=parsed,
                    step_score=step_score,
                    llm_meta=meta,  # ✅ store LLM meta
                )
                br.steps.append(step)

                if parsed:
                    current_graph = aggregate_step_to_graph(current_graph, parsed)
                cumulative_score += step_score

                if parsed is not None and not parsed.get("nodes") and not parsed.get("relations"):
                    break

            if canonicalize:
                current_graph = canonicalize_nodes_relations(current_graph, perception=perception)

            br.final_graph = current_graph
            br.loglik = cumulative_score
            branches.append(br)

        branches.sort(key=lambda x: x.loglik, reverse=True)
        return branches
