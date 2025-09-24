"""
Module: src/tot/perception_orchestrator.py
Purpose: Orchestrate Tree-of-Thought reasoning with perception functions.
Each branch = one perception applied across all chunks.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import logging

from Hackathon_ECI_SPRNS.venv.ToT.perceptions import list_perceptions, get_perception
from Hackathon_ECI_SPRNS.venv.ToT.orchestrator import ToTOrchestrator, BranchResult, aggregate_step_to_graph
from Hackathon_ECI_SPRNS.venv.Schema.canonicalizer import canonicalize_nodes_relations

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PerceptionOrchestrator:
    def __init__(self, llm_client, max_steps: int = 5, max_add_per_step: int = 4):
        self.base_orch = ToTOrchestrator(llm_client, max_steps, max_add_per_step)

    def run(
            self,
            chunks: List[Dict[str, Any]],
            perceptions: Optional[List[str]] = None,
            max_branches_per_perception: int = 1,
            temperature: float = 0.0,
    ) -> Dict[str, BranchResult]:
        if perceptions is None:
            perceptions = list_perceptions().keys()
        results: Dict[str, BranchResult] = {}

        for pname in perceptions:
            logger.info("=== Starting perception: %s ===", pname)
            perception = get_perception(pname)

            combined_graph, combined_steps, total_score = {"nodes": [], "relations": []}, [], 0.0

            for chunk in chunks:
                branches = self.base_orch.generate_branches(
                    chunk=chunk,
                    perception=pname,
                    max_branches=max_branches_per_perception,
                    temperature=temperature,
                )
                if branches:
                    best = branches[0]

                    # ✅ Canonicalize the graph before merging
                    canon_graph = canonicalize_nodes_relations(best.final_graph, perception=pname)
                    best.final_graph = canon_graph  # overwrite with canonicalized version

                    combined_graph = aggregate_step_to_graph(combined_graph, canon_graph)
                    combined_steps.extend(best.steps)
                    total_score += best.loglik

            br = BranchResult(
                branch_id=f"perception_{pname}",
                chunk_id="ALL",
                perception=pname,
                steps=combined_steps,
                final_graph=combined_graph,  # ✅ already canonicalized
                loglik=total_score,
                provenance={
                    "mode": "perception",
                    "schema_focus": perception.schema_focus,
                },
            )
            results[pname] = br

        return results


def merge_perception_graphs(branches: Dict[str, BranchResult]) -> Dict[str, Any]:
    merged_graph = {"nodes": [], "relations": []}
    for br in branches.values():
        merged_graph = aggregate_step_to_graph(merged_graph, br.final_graph)
    return merged_graph
