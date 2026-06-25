"""
Random Forest scoring head on TGN embeddings (NID-TGN, SPACE 2024).

Fitted post-training on validation embeddings. At inference receives
concatenated [z_src, z_dst, edge_features] and outputs fraud probability
with feature importances for explainability.

This is the ensemble's own copy — tgn_learn/ is read-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class RFScoringResult:
    """Result from the RF scoring head."""

    fraud_probability: float
    feature_importances: np.ndarray
    feature_names: list[str]


class RFScoringHead:
    """Random Forest scoring head on TGN embeddings.

    Handles class imbalance via class_weight='balanced'.

    Args:
        n_estimators: Number of trees
        max_depth: Max tree depth (None for unlimited)
        min_samples_leaf: Min samples per leaf
        random_state: Seed for reproducibility
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ):
        from sklearn.ensemble import RandomForestClassifier

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
        self._is_fitted = False
        self._feature_names: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list[str]] = None) -> "RFScoringHead":
        if len(X) == 0:
            raise ValueError("Cannot fit on empty data")
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")

        n_features = X.shape[1]
        if feature_names is not None:
            if len(feature_names) != n_features:
                raise ValueError(f"feature_names length ({len(feature_names)}) != n_features ({n_features})")
            self._feature_names = feature_names
        else:
            self._feature_names = [f"feat_{i}" for i in range(n_features)]

        self.rf.fit(X, y)
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("RFScoringHead not fitted")
        return self.rf.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int32)

    def score_single(self, x: np.ndarray) -> RFScoringResult:
        if not self._is_fitted:
            raise RuntimeError("RFScoringHead not fitted")
        x_2d = x.reshape(1, -1)
        prob = self.rf.predict_proba(x_2d)[0, 1]
        return RFScoringResult(
            fraud_probability=float(prob),
            feature_importances=self.rf.feature_importances_,
            feature_names=self._feature_names,
        )

    def get_feature_importances(self) -> dict[str, float]:
        if not self._is_fitted:
            raise RuntimeError("RFScoringHead not fitted")
        named = dict(zip(self._feature_names, self.rf.feature_importances_))
        return dict(sorted(named.items(), key=lambda kv: -kv[1]))

    def get_top_features(self, n: int = 5) -> list[tuple[str, float]]:
        return list(self.get_feature_importances().items())[:n]

    def save(self, path: str | Path) -> None:
        import joblib
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted RFScoringHead")
        joblib.dump({
            "rf": self.rf, "feature_names": self._feature_names,
            "n_estimators": self.n_estimators, "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf, "random_state": self.random_state,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "RFScoringHead":
        import joblib
        state = joblib.load(path)
        head = cls(
            n_estimators=state["n_estimators"], max_depth=state["max_depth"],
            min_samples_leaf=state["min_samples_leaf"], random_state=state["random_state"],
        )
        head.rf = state["rf"]
        head._feature_names = state["feature_names"]
        head._is_fitted = True
        return head

    @staticmethod
    def build_feature_names(embedding_dim: int, edge_feat_dim: int) -> list[str]:
        names = [f"z_src_{i}" for i in range(embedding_dim)]
        names += [f"z_dst_{i}" for i in range(embedding_dim)]
        names += [f"edge_feat_{i}" for i in range(edge_feat_dim)]
        return names
