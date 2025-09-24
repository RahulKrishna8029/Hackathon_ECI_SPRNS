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

from Hackathon_ECI_SPRNS.venv.ToT.llm_client import LLMClient  # <-- import actual client abstraction

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
    if not text or not text.strip():
        return None
    try:
        text_strip = text.strip()
        if text_strip[0] in ["{", "["]:
            return json.loads(text_strip)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
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
        c = n.get("confidence", 0.5)
        c = max(min(float(c), 0.999999), 1e-6)
        total += math.log(c)
        count += 1
    for r in parsed_json.get("relations", []):
        c = r.get("confidence", 0.5)
        c = max(min(float(c), 0.999999), 1e-6)
        total += math.log(c)
        count += 1
    return total if count > 0 else -2.0


# ---------------------------
# Prompt templates
# ---------------------------

ROOT_PROMPT_TEMPLATE = """You are a Knowledge-Graph extractor.
Perception: {perception}
Chunk text:
\"\"\"{chunk_text}\"\"\"

TASK:
Return JSON with arrays "nodes" and "relations".
Each node: {{ "label": "...", "span": "...", "confidence": 0-1 }}
Each relation: {{ "from": "...", "to": "...", "type": "...", "confidence": 0-1 }}
If no new additions, return {{ "nodes": [], "relations": [] }}.
"""

STEP_PROMPT_TEMPLATE = """Continue Knowledge Graph extraction.
Perception: {perception}
Chunk text:
\"\"\"{chunk_text}\"\"\"

Partial graph: {partial_graph}

TASK: Propose up to {max_add} new nodes/relations in JSON.
If no additions, return {{ "nodes": [], "relations": [] }}.
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
            temperature: float = 0.0
    ) -> List[BranchResult]:
        branches: List[BranchResult] = []
        for b in range(max_branches):
            branch_id = f"{chunk['chunk_id']}_b{b}_{uuid.uuid4().hex[:8]}"
            br = BranchResult(
                branch_id=branch_id,
                chunk_id=chunk["chunk_id"],
                perception=perception,
                provenance={"chunk_meta": {k: chunk.get(k) for k in ("doc_id", "entity_id", "start_char", "end_char")},
                            "branch_seed": b},
            )
            current_graph, cumulative_score = {"nodes": [], "relations": []}, 0.0
            for step_idx in range(self.max_steps):
                if step_idx == 0:
                    prompt = ROOT_PROMPT_TEMPLATE.format(perception=(perception or "general"),
                                                         chunk_text=chunk["raw_text"])
                else:
                    prompt = STEP_PROMPT_TEMPLATE.format(perception=(perception or "general"),
                                                         chunk_text=chunk["raw_text"],
                                                         partial_graph=json.dumps(current_graph),
                                                         max_add=self.max_add_per_step)
                completion_text, meta = self.llm.complete(prompt, temperature=temperature)
                parsed = safe_parse_json(completion_text)
                step_score = compute_step_score(parsed)
                step = StepResult(step_index=step_idx, prompt=prompt,
                                  completion_text=completion_text, parsed_json=parsed,
                                  step_score=step_score, llm_meta=meta)
                br.steps.append(step)
                if parsed:
                    current_graph = aggregate_step_to_graph(current_graph, parsed)
                cumulative_score += step_score
                if parsed is not None and not parsed.get("nodes") and not parsed.get("relations"):
                    break
            br.final_graph, br.loglik = current_graph, cumulative_score
            branches.append(br)
        branches.sort(key=lambda x: x.loglik, reverse=True)
        return branches