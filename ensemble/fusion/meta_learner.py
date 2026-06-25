"""
LightGBM Meta-Learner — stacking ensemble over detector scores.

Takes all detector scores + raw transaction features as input,
outputs calibrated fraud probability. Trained on chronological
holdout after all detectors are individually fitted.

Why LightGBM over simple averaging:
- Each detector has complementary failure modes
- Learns that flow_dag=0.8 + tgn=0.3 ≠ tgn=0.8 + flow_dag=0.3
- Handles heterogeneous score scales
- Feature importances show which detector matters for which fraud type
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class MetaLearnerResult:
    """Result from the meta-learner."""

    fraud_probability: float
    detector_contributions: dict[str, float]


class EnsembleMetaLearner:
    """LightGBM meta-learner over detector scores (stacking ensemble).

    Input features per transaction:
      - detector_scores: [tgn_score, rf_score, flow_dag_score,
                          semantic_score, drift_score]  (5 floats)
      - raw_features: transaction features (variable dim)

    Args:
        n_estimators: Number of boosting rounds
        learning_rate: Step size for each round
        max_depth: Max tree depth
        random_state: Seed
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = None
        self._feature_names: list[str] = []
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        detector_scores: np.ndarray,
        raw_features: np.ndarray,
        labels: np.ndarray,
        detector_names: Optional[list[str]] = None,
    ) -> "EnsembleMetaLearner":
        """Train the meta-learner on detector outputs + raw features.

        Args:
            detector_scores: [n_samples, n_detectors]
            raw_features: [n_samples, n_raw_features]
            labels: [n_samples] binary fraud labels
            detector_names: Names for detector score columns

        Returns:
            self (for chaining)
        """
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            # Fallback to sklearn GradientBoosting if lightgbm unavailable
            from sklearn.ensemble import GradientBoostingClassifier as LGBMClassifier
            self._using_sklearn_fallback = True

        X = np.hstack([detector_scores, raw_features])

        # Build feature names
        n_detectors = detector_scores.shape[1]
        if detector_names and len(detector_names) == n_detectors:
            det_names = [f"det_{name}" for name in detector_names]
        else:
            det_names = [f"det_{i}" for i in range(n_detectors)]

        n_raw = raw_features.shape[1]
        raw_names = [f"raw_{i}" for i in range(n_raw)]
        self._feature_names = det_names + raw_names

        # Fit
        if hasattr(self, '_using_sklearn_fallback') and self._using_sklearn_fallback:
            self._model = LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
        else:
            self._model = LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                class_weight="balanced",
                verbose=-1,
            )

        self._model.fit(X, labels)
        self._fitted = True
        return self

    def predict_proba(
        self,
        detector_scores: np.ndarray,
        raw_features: np.ndarray,
    ) -> np.ndarray:
        """Predict fraud probabilities.

        Args:
            detector_scores: [n_samples, n_detectors]
            raw_features: [n_samples, n_raw_features]

        Returns:
            [n_samples] fraud probabilities
        """
        if not self._fitted:
            raise RuntimeError("Meta-learner not fitted")

        X = np.hstack([detector_scores, raw_features])
        return self._model.predict_proba(X)[:, 1]

    def predict_single(
        self,
        detector_scores: np.ndarray,
        raw_features: np.ndarray,
        detector_names: Optional[list[str]] = None,
    ) -> MetaLearnerResult:
        """Predict for a single transaction with contribution breakdown.

        Args:
            detector_scores: [n_detectors] scores from each detector
            raw_features: [n_raw] raw transaction features
            detector_names: Names for contribution dict

        Returns:
            MetaLearnerResult with probability and contributions
        """
        if not self._fitted:
            raise RuntimeError("Meta-learner not fitted")

        X = np.concatenate([detector_scores, raw_features]).reshape(1, -1)
        prob = float(self._model.predict_proba(X)[0, 1])

        # Estimate contributions from feature importances
        contributions = {}
        if detector_names and hasattr(self._model, 'feature_importances_'):
            importances = self._model.feature_importances_
            n_det = len(detector_names)
            det_importance_sum = importances[:n_det].sum() + 1e-8
            for i, name in enumerate(detector_names):
                contributions[name] = float(
                    importances[i] / det_importance_sum * detector_scores[i]
                )

        return MetaLearnerResult(
            fraud_probability=prob,
            detector_contributions=contributions,
        )

    def feature_importances(self) -> dict[str, float]:
        """Get named feature importances."""
        if not self._fitted or not hasattr(self._model, 'feature_importances_'):
            return {}
        named = dict(zip(self._feature_names, self._model.feature_importances_))
        return dict(sorted(named.items(), key=lambda kv: -kv[1]))

    def save(self, path: str | Path) -> None:
        """Save meta-learner to disk."""
        import joblib
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted meta-learner")
        joblib.dump({
            "model": self._model,
            "feature_names": self._feature_names,
            "params": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
            },
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleMetaLearner":
        """Load from disk."""
        import joblib
        state = joblib.load(path)
        ml = cls(**state["params"])
        ml._model = state["model"]
        ml._feature_names = state["feature_names"]
        ml._fitted = True
        return ml
