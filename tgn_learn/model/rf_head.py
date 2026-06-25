"""
Random Forest scoring head on TGN embeddings (NID-TGN, SPACE 2024).

After TGN training produces node embeddings, this RF head is fitted on
the validation set to produce calibrated fraud probabilities. At inference
it receives concatenated [z_src, z_dst, edge_features] and outputs a
fraud probability with feature importances for explainability.

Why RF over the MLP LinkPredictor for final scoring:
    - Handles class imbalance natively via class_weight='balanced'
    - Produces feature importances → direct explainability
    - No resampling needed, preserving temporal ordering
    - Ensemble of decision trees is more robust to distribution shift

The TGN still trains with MLP heads for the contrastive loss. This RF
is the *inference-time* scoring head fitted post-training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class RFScoringResult:
    """Result from the RF scoring head.

    Attributes:
        fraud_probability: Calibrated fraud probability in [0, 1]
        feature_importances: Per-feature importance scores (Gini)
        feature_names: Names corresponding to importance scores
    """

    fraud_probability: float
    feature_importances: np.ndarray
    feature_names: list[str]


class RFScoringHead:
    """Random Forest scoring head on TGN embeddings.

    Fitted after TGN training on the validation set. At inference,
    receives concatenated [z_src, z_dst, edge_features] and outputs
    fraud probability with feature importances.

    Handles class imbalance via class_weight='balanced' — no resampling
    needed, which preserves temporal ordering.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth (None for unlimited).
        min_samples_leaf: Minimum samples per leaf node.
        random_state: Random seed for reproducibility.

    Example:
        >>> head = RFScoringHead(n_estimators=200)
        >>> # X = concat([z_src, z_dst, edge_features]) from val set
        >>> head.fit(X_val, y_val)
        >>> probs = head.predict_proba(X_test)
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
        """Whether the RF has been fitted on data."""
        return self._is_fitted

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "RFScoringHead":
        """Fit the RF on TGN embeddings + edge features.

        Args:
            X: Feature matrix [n_samples, n_features].
                Expected layout: [z_src (emb_dim), z_dst (emb_dim), edge_feats (feat_dim)]
            y: Binary labels [n_samples] (0=legit, 1=fraud)
            feature_names: Optional names for each feature column.
                If not provided, generates default names.

        Returns:
            self (for chaining)
        """
        if len(X) == 0:
            raise ValueError("Cannot fit RF head on empty data")
        if len(X) != len(y):
            raise ValueError(
                f"X and y length mismatch: {len(X)} vs {len(y)}"
            )

        n_features = X.shape[1]
        if feature_names is not None:
            if len(feature_names) != n_features:
                raise ValueError(
                    f"feature_names length ({len(feature_names)}) != "
                    f"n_features ({n_features})"
                )
            self._feature_names = feature_names
        else:
            self._feature_names = [f"feat_{i}" for i in range(n_features)]

        self.rf.fit(X, y)
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud probabilities.

        Args:
            X: Feature matrix [n_samples, n_features]

        Returns:
            Fraud probabilities [n_samples] in [0, 1]
        """
        if not self._is_fitted:
            raise RuntimeError("RFScoringHead has not been fitted yet")
        # predict_proba returns [n_samples, 2] for binary classification
        return self.rf.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary fraud labels.

        Args:
            X: Feature matrix [n_samples, n_features]
            threshold: Decision threshold (default 0.5)

        Returns:
            Binary predictions [n_samples]
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int32)

    def score_single(self, x: np.ndarray) -> RFScoringResult:
        """Score a single transaction with feature importances.

        Args:
            x: Feature vector [n_features] for one transaction

        Returns:
            RFScoringResult with probability and per-feature importances
        """
        if not self._is_fitted:
            raise RuntimeError("RFScoringHead has not been fitted yet")

        x_2d = x.reshape(1, -1)
        prob = self.rf.predict_proba(x_2d)[0, 1]

        return RFScoringResult(
            fraud_probability=float(prob),
            feature_importances=self.rf.feature_importances_,
            feature_names=self._feature_names,
        )

    def get_feature_importances(self) -> dict[str, float]:
        """Get named feature importances (Gini importance).

        Returns:
            Dict mapping feature name → importance score.
            Sorted by importance (descending).
        """
        if not self._is_fitted:
            raise RuntimeError("RFScoringHead has not been fitted yet")

        importances = self.rf.feature_importances_
        named = dict(zip(self._feature_names, importances))
        return dict(sorted(named.items(), key=lambda kv: -kv[1]))

    def get_top_features(self, n: int = 5) -> list[tuple[str, float]]:
        """Get the top-N most important features.

        Args:
            n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples, descending
        """
        all_importances = self.get_feature_importances()
        return list(all_importances.items())[:n]

    def save(self, path: str | Path) -> None:
        """Save the fitted RF model to disk.

        Args:
            path: File path (will use joblib serialisation)
        """
        import joblib

        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted RFScoringHead")

        state = {
            "rf": self.rf,
            "feature_names": self._feature_names,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str | Path) -> "RFScoringHead":
        """Load a fitted RF model from disk.

        Args:
            path: Path to saved model file

        Returns:
            Fitted RFScoringHead instance
        """
        import joblib

        state = joblib.load(path)
        head = cls(
            n_estimators=state["n_estimators"],
            max_depth=state["max_depth"],
            min_samples_leaf=state["min_samples_leaf"],
            random_state=state["random_state"],
        )
        head.rf = state["rf"]
        head._feature_names = state["feature_names"]
        head._is_fitted = True
        return head

    @staticmethod
    def build_feature_names(
        embedding_dim: int, edge_feat_dim: int
    ) -> list[str]:
        """Generate standard feature names for the RF input layout.

        The expected input layout is:
            [z_src_0, ..., z_src_{d-1}, z_dst_0, ..., z_dst_{d-1}, ef_0, ..., ef_{k-1}]

        Args:
            embedding_dim: Dimension of source/destination embeddings
            edge_feat_dim: Dimension of edge feature vector

        Returns:
            List of feature names
        """
        names = []
        names.extend([f"z_src_{i}" for i in range(embedding_dim)])
        names.extend([f"z_dst_{i}" for i in range(embedding_dim)])
        names.extend([f"edge_feat_{i}" for i in range(edge_feat_dim)])
        return names
