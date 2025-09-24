"""
Module: src/tot/perceptions.py
Purpose: Define candidate Perception Functions (π_k) for KG extraction.
Improvements:
 - schema validation
 - optional versioning
 - helpers to obtain prompt-ready schema contexts
"""

from typing import Dict, Any, List
import datetime

class Perception:
    """
    Encapsulates a perception function π_k.
    Stores schema focus (nodes/relations) and description.
    """

    def __init__(self, name: str, description: str, schema_focus: Dict[str, List[str]], version: str = "v1"):
        if "nodes" not in schema_focus or "relations" not in schema_focus:
            raise ValueError("schema_focus must contain 'nodes' and 'relations' lists.")
        self.name = name
        self.description = description
        self.schema_focus = {"nodes": list(schema_focus["nodes"]), "relations": list(schema_focus["relations"])}
        self.version = version
        self.created_at = datetime.datetime.utcnow().isoformat()

    def to_prompt_context(self) -> str:
        """
        Return a compact text block describing schema focus,
        usable in prompt templates for the LLM.
        """
        nodes = ", ".join(self.schema_focus.get("nodes", []))
        rels = ", ".join(self.schema_focus.get("relations", []))
        return (
            f"Perception: {self.name} (schema version: {self.version})\n"
            f"Description: {self.description}\n"
            f"Only attempt to extract these node labels: {nodes}\n"
            f"And these relation types: {rels}\n"
            f"If none apply, return empty arrays."
        )

    def __repr__(self) -> str:
        return f"Perception(name={self.name}, nodes={len(self.schema_focus['nodes'])}, rels={len(self.schema_focus['relations'])}, version={self.version})"


# ---------------------------
# Candidate Perceptions
# ---------------------------

PERCEPTIONS: Dict[str, Perception] = {
    "identity": Perception(
        name="identity",
        description="Capture personal identifiers.",
        schema_focus={
            "nodes": ["Name", "DOB", "Address", "Phone", "Email"],
            "relations": ["updated_to", "verified_by"],
        },
        version="v1",
    ),
    "transaction": Perception(
        name="transaction",
        description="Capture transactional history.",
        schema_focus={
            "nodes": ["Transaction", "Account", "Merchant", "Timestamp"],
            "relations": ["made_by", "sent_to", "received_from"],
        },
        version="v1",
    ),
    "behavioral": Perception(
        name="behavioral",
        description="Model user actions & patterns.",
        schema_focus={
            "nodes": ["Device", "Channel", "Location"],
            "relations": ["accessed_from", "logged_in", "attempted_at"],
        },
        version="v1",
    ),
    "relational": Perception(
        name="relational",
        description="Extract social/professional ties.",
        schema_focus={
            "nodes": ["Customer", "Employer", "Family", "Partner"],
            "relations": ["employed_at", "related_to", "authorized_by"],
        },
        version="v1",
    ),
    "risk": Perception(
        name="risk",
        description="Focus on fraud, AML, and compliance.",
        schema_focus={
            "nodes": ["Transaction", "WatchlistEntity", "Rule"],
            "relations": ["flagged_by", "violates", "linked_to"],
        },
        version="v1",
    ),
    "temporal": Perception(
        name="temporal",
        description="Focus on historical consistency.",
        schema_focus={
            "nodes": ["Event", "Timestamp"],
            "relations": ["preceded_by", "updated_at", "valid_until"],
        },
        version="v1",
    ),
    "semantic": Perception(
        name="semantic",
        description="Capture conceptual/ontological meaning.",
        schema_focus={
            "nodes": ["Concept", "Category", "Document"],
            "relations": ["is_a", "refers_to", "describes"],
        },
        version="v1",
    ),
}


# ---------------------------
# Utilities
# ---------------------------

def get_perception(name: str) -> Perception:
    """Retrieve a perception object by name."""
    if name not in PERCEPTIONS:
        raise ValueError(f"Unknown perception '{name}'. Available: {list(PERCEPTIONS.keys())}")
    return PERCEPTIONS[name]


def list_perceptions() -> Dict[str, Perception]:
    """Return dict of all available perceptions."""
    return PERCEPTIONS


def list_perception_names() -> List[str]:
    """Return list of all perception names."""
    return list(PERCEPTIONS.keys())


def list_perception_contexts() -> Dict[str, str]:
    """Return dict {perception_name: schema_context_str} for all perceptions."""
    return {name: p.to_prompt_context() for name, p in PERCEPTIONS.items()}
