# src/proj/projection.py
"""
Semantic Subspace Learner (Projection P_S)

Improvements:
 - fragment-aware APIs (project_fragment, project_batch)
 - prefer fragment.phi_emb (LLM embedding) with fallback to phi_raw
 - vectorized batch projection when possible
 - robust save/load (JSON) and metadata
 - optional PyTorch learnable model preserved (experimental)
 - helpers to infer dimension from sample fragments and fit directly from fragments
"""

from __future__ import annotations
import os
import json
import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from sklearn.decomposition import IncrementalPCA
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Optional torch support
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# -------------------------
# Utilities
# -------------------------

def orthonormalize_columns(mat: np.ndarray) -> np.ndarray:
    """Return orthonormal basis for columns of mat using QR or SVD (d x k)."""
    try:
        q, r = np.linalg.qr(mat)
        return q[:, :mat.shape[1]]
    except Exception:
        U, _, _ = np.linalg.svd(mat, full_matrices=False)
        return U[:, :mat.shape[1]]


def subspace_angle(U_old: Optional[np.ndarray], U_new: Optional[np.ndarray]) -> float:
    """
    Davis–Kahan-like subspace angle (spectral norm of sin theta).
    Returns scalar in [0,1], smaller is better (0 means identical).
    """
    if U_old is None or U_new is None:
        return 1.0
    C = U_old.T @ U_new
    s = np.linalg.svd(C, compute_uv=False)
    sin_vals = np.sqrt(np.clip(1.0 - s**2, 0.0, 1.0))
    return float(np.max(sin_vals))


# -------------------------
# PCA-based Projection Model
# -------------------------

class ProjectionModelPCA:
    """
    Principal Components based semantic subspace using IncrementalPCA.

    You can construct with d specified, or call `infer_dim_from_fragments` /
    `fit_from_fragments` to build from sample fragments (recommended).
    """

    def __init__(self, d: Optional[int] = None, k: int = 32, batch_size: int = 256,
                 perceptions: Optional[List[str]] = None):
        if d is not None:
            assert k < d, "k must be less than embedding dimension d"
        self.d = int(d) if d is not None else None
        self.k = int(k)
        self.batch_size = int(batch_size)
        # Delay creating ipca until d is known
        self.ipca: Optional[IncrementalPCA] = IncrementalPCA(n_components=self.k, batch_size=self.batch_size) if self.d else None
        self._fitted = False
        self.U: Optional[np.ndarray] = None  # d x k
        self.perceptions = perceptions or []
        self.created_at = datetime.utcnow().isoformat()

    # -------------------------
    # Helpers to infer dims from fragments
    # -------------------------

    @staticmethod
    def infer_dim_from_fragments(fragments: List[Any]) -> int:
        """
        Inspect a list of GraphFragment objects and return embedding dimension
        based on phi_emb (preferred) or phi_raw.
        """
        for f in fragments:
            if getattr(f, "phi_emb", None) is not None:
                return int(np.asarray(f.phi_emb).reshape(-1).shape[0])
            if getattr(f, "phi_raw", None) is not None:
                return int(np.asarray(f.phi_raw).reshape(-1).shape[0])
        raise ValueError("No fragment contains phi_emb or phi_raw to infer dimension")

    def initialize_ipca_if_needed(self):
        if self.d is None:
            raise RuntimeError("Projector dimension d is not set. Call fit_from_fragments or set d in constructor.")
        if self.ipca is None:
            self.ipca = IncrementalPCA(n_components=self.k, batch_size=self.batch_size)

    # -------------------------
    # Fitting
    # -------------------------

    def fit_initial(self, embeddings: np.ndarray):
        """Fit initial PCA on embeddings (N x d)."""
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if self.d is None:
            self.d = embeddings.shape[1]
            self.initialize_ipca_if_needed()
        if embeddings.ndim != 2 or embeddings.shape[1] != self.d:
            raise ValueError(f"embeddings must be shape (N, {self.d}); got {embeddings.shape}")
        logger.info("Fitting initial IncrementalPCA on %d samples (d=%d k=%d)", embeddings.shape[0], self.d, self.k)
        self.ipca.partial_fit(embeddings)
        comps = self.ipca.components_.T  # d x k
        self.U = orthonormalize_columns(comps)
        self._fitted = True

    def fit_incremental(self, embeddings: np.ndarray):
        """Incremental update using a batch of embeddings (N x d)."""
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if self.d is None:
            raise ValueError("Model dimension 'd' unknown. Fit initial first or set d explicitly.")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.d:
            raise ValueError(f"embeddings must be shape (N, {self.d}); got {embeddings.shape}")
        logger.info("Incremental PCA update with %d samples", embeddings.shape[0])
        self.ipca.partial_fit(embeddings)
        comps = self.ipca.components_.T
        self.U = orthonormalize_columns(comps)
        self._fitted = True

    def fit_from_fragments(self, fragments: List[Any], normalize: bool = True):
        """
        Convenience: extract embeddings from fragments and fit PCA.
        Accepts list of GraphFragment; prefers phi_emb, falls back to phi_raw.
        """
        if not fragments:
            raise ValueError("No fragments provided")
        d_infer = self.infer_dim_from_fragments(fragments)
        self.d = d_infer
        self.ipca = IncrementalPCA(n_components=self.k, batch_size=self.batch_size)
        vecs = []
        for f in fragments:
            if getattr(f, "phi_emb", None) is not None:
                e = np.asarray(f.phi_emb, dtype=float).reshape(-1)
            elif getattr(f, "phi_raw", None) is not None:
                e = np.asarray(f.phi_raw, dtype=float).reshape(-1)
            else:
                continue
            if e.shape[0] != self.d:
                raise ValueError(f"Fragment embedding dim {e.shape[0]} inconsistent with inferred d {self.d}")
            if normalize:
                nrm = np.linalg.norm(e)
                if nrm > 0:
                    e = e / nrm
            vecs.append(e)
        E = np.vstack(vecs).astype(np.float64)
        self.fit_initial(E)

    # -------------------------
    # Projection helpers
    # -------------------------

    def _check_fitted(self):
        if not self._fitted or self.U is None:
            raise RuntimeError("ProjectionModelPCA not fitted yet")

    def project(self, e: np.ndarray) -> np.ndarray:
        """Project vector e (d,) onto semantic subspace: P_S e = U U^T e."""
        self._check_fitted()
        e = np.asarray(e, dtype=float).reshape(-1)
        if e.shape[0] != self.d:
            raise ValueError(f"Input vector dim {e.shape[0]} != model dim {self.d}")
        return self.U @ (self.U.T @ e)

    def residual(self, e: np.ndarray) -> np.ndarray:
        self._check_fitted()
        e = np.asarray(e, dtype=float).reshape(-1)
        return e - self.project(e)

    def project_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project a batch of embeddings (N x d) using matrix ops (vectorized).
        Returns (N x d) projected vectors.
        """
        self._check_fitted()
        E = np.asarray(embeddings, dtype=float)
        if E.ndim != 2 or E.shape[1] != self.d:
            raise ValueError(f"embeddings must be shape (N, {self.d})")
        coords = E @ self.U            # (N x k)
        proj = coords @ self.U.T      # (N x d)
        return proj

    # -------------------------
    # Fragment-aware
    # -------------------------

    @staticmethod
    def _embedding_for_fragment(fragment) -> np.ndarray:
        """
        Prefer fragment.phi_emb (LLM embedding), fallback to phi_raw (must be of correct dim).
        Returns 1D numpy array.
        """
        if getattr(fragment, "phi_emb", None) is not None:
            e = np.asarray(getattr(fragment, "phi_emb"), dtype=float).reshape(-1)
        elif getattr(fragment, "phi_raw", None) is not None:
            e = np.asarray(fragment.phi_raw, dtype=float).reshape(-1)
        else:
            raise ValueError("Fragment has neither 'phi_emb' nor 'phi_raw' available")
        return e

    def project_fragment(self, fragment) -> Dict[str, Any]:
        """
        Project a GraphFragment into subspace + residual, with metadata.
        Accepts fragment objects with phi_emb OR phi_raw.
        """
        e = self._embedding_for_fragment(fragment)
        if self.d is None:
            raise RuntimeError("Projector dimension unknown; call fit_from_fragments() or set d.")
        if e.shape[0] != self.d:
            raise ValueError(f"Fragment embedding dim {e.shape[0]} != projector dim {self.d}")
        proj = self.project(e)
        resid = e - proj
        return {
            "fragment_id": getattr(fragment, "fragment_id", None),
            "entity_id": getattr(fragment, "entity_id", None),
            "perception": fragment.provenance.get("perception") if getattr(fragment, "provenance", None) else None,
            "proj_vector": proj.tolist(),
            "resid_vector": resid.tolist(),
        }

    def project_fragments_bulk(self, fragments: List[Any]) -> List[Dict[str, Any]]:
        """
        Efficiently project a list of fragments by collecting embeddings, projecting vectorized,
        and returning per-fragment dicts.
        """
        if not fragments:
            return []
        Es = []
        ids = []
        for f in fragments:
            e = self._embedding_for_fragment(f)
            if self.d is None:
                self.d = e.shape[0]
            if e.shape[0] != self.d:
                raise ValueError(f"Fragment {getattr(f,'fragment_id',None)} embedding dim {e.shape[0]} != {self.d}")
            Es.append(e)
            ids.append(getattr(f, "fragment_id", None))
        E = np.vstack(Es)  # (N x d)
        P = self.project_batch(E)  # (N x d)
        R = E - P
        out = []
        for i, f in enumerate(fragments):
            out.append({
                "fragment_id": ids[i],
                "entity_id": getattr(f, "entity_id", None),
                "perception": f.provenance.get("perception") if getattr(f, "provenance", None) else None,
                "proj_vector": P[i].tolist(),
                "resid_vector": R[i].tolist()
            })
        return out

    # -------------------------
    # Persistence
    # -------------------------

    def save(self, path: str):
        """Save to JSON (U as nested list) along with metadata."""
        if self.U is None:
            raise RuntimeError("Nothing to save: model U not set")
        obj = {
            "d": int(self.d),
            "k": int(self.k),
            "U": self.U.tolist(),
            "perceptions": self.perceptions,
            "created_at": self.created_at,
            "saved_at": datetime.utcnow().isoformat(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        logger.info("Saved ProjectionModelPCA to %s", path)

    @classmethod
    def load(cls, path: str) -> "ProjectionModelPCA":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        model = cls(d=int(obj["d"]), k=int(obj["k"]), perceptions=obj.get("perceptions", []))
        if obj.get("U") is not None:
            model.U = np.array(obj["U"], dtype=float)
            model._fitted = True
            model.ipca = IncrementalPCA(n_components=model.k, batch_size=model.batch_size)
        return model


# -------------------------
# Learnable Projection Model (contrastive) — EXPERIMENTAL
# -------------------------

if TORCH_AVAILABLE:
    class ProjectionModelLearnable(nn.Module):
        """Experimental learnable projection via InfoNCE contrastive loss."""

        def __init__(self, d: int, k: int = 32, temp: float = 0.07, lr: float = 1e-3, device: str = "cpu"):
            super().__init__()
            self.d = d
            self.k = k
            self.temp = temp
            self.device = torch.device(device)
            self.U_param = nn.Parameter(torch.randn(d, k, device=self.device) * 0.01)
            self._optimizer = optim.Adam([self.U_param], lr=lr)
            logger.warning("ProjectionModelLearnable is experimental — requires positive pairs for training.")

        def orthonormalize_param(self, U_param: torch.Tensor) -> torch.Tensor:
            try:
                U, _, _ = torch.linalg.svd(U_param, full_matrices=False)
                return U[:, :U_param.shape[1]]
            except Exception:
                Q, _ = torch.linalg.qr(U_param)
                return Q[:, :U_param.shape[1]]

        def forward_project(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            U = self.orthonormalize_param(self.U_param)
            z = torch.matmul(e, U)  # (B, k)
            proj = torch.matmul(z, U.t())  # (B, d)
            return proj, z

        def info_nce_loss(self, z_q: torch.Tensor, z_pos: torch.Tensor, z_all: torch.Tensor) -> torch.Tensor:
            z_q = nn.functional.normalize(z_q, dim=1)
            logits_all = torch.matmul(z_q, z_all.t()) / self.temp
            labels = torch.arange(z_q.size(0), device=z_q.device)
            return nn.CrossEntropyLoss()(logits_all, labels)

        def step_update(self, batch: Dict[str, torch.Tensor]):
            e_q, e_pos = batch["e_q"].to(self.device), batch["e_pos"].to(self.device)
            _, z_q = self.forward_project(e_q)
            _, z_pos = self.forward_project(e_pos)
            z_all = torch.cat([z_q, z_pos], dim=0)
            loss = self.info_nce_loss(z_q, z_pos, z_all)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            return loss.item()

        def save(self, path: str):
            torch.save(self.state_dict(), path)

        def load_state(self, path: str):
            self.load_state_dict(torch.load(path, map_location=self.device))

else:
    ProjectionModelLearnable = None


# -------------------------
# Evaluation helpers
# -------------------------

def projection_invariance_rate(pairs: List[Tuple[np.ndarray, np.ndarray]],
                               projector: Any,
                               eps: float = 1e-3) -> float:
    """Fraction of same-entity pairs with projected distance < eps."""
    if not pairs:
        return 0.0
    ok = 0
    for e1, e2 in pairs:
        p1, p2 = projector.project(e1), projector.project(e2)
        if np.linalg.norm(p1 - p2) < eps:
            ok += 1
    return ok / len(pairs)


def evaluate_projection(pairs: List[Tuple[np.ndarray, np.ndarray]],
                        projector: Any,
                        eps: float = 1e-3) -> Dict[str, float]:
    """Compute multiple projection metrics on same-entity pairs."""
    if not pairs:
        return {"PIR": 0.0, "mean_proj_disp": 0.0}
    pir = projection_invariance_rate(pairs, projector, eps)
    disp = np.mean([np.linalg.norm(projector.project(e1) - projector.project(e2))
                    for e1, e2 in pairs])
    return {"PIR": pir, "mean_proj_disp": float(disp)}