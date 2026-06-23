"""Fraud detection evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class FraudMetrics:
    """
    Evaluation metrics for fraud detection.

    Collects predictions and computes standard fraud detection metrics
    including AUC-PR (the primary metric for imbalanced fraud data).
    """

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_pr: float = 0.0
    auc_roc: float = 0.0

    @classmethod
    def compute(cls, scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> "FraudMetrics":
        """Compute all metrics from raw scores and labels.

        Args:
            scores: Predicted fraud probabilities [n_samples]
            labels: Ground truth labels (0 or 1) [n_samples]
            threshold: Classification threshold for precision/recall/F1

        Returns:
            FraudMetrics instance with all metrics computed
        """
        # Filter out unlabeled samples (label == -1)
        mask = labels >= 0
        if mask.sum() == 0:
            return cls()

        scores = scores[mask]
        labels = labels[mask]

        # Binary predictions at threshold
        preds = (scores >= threshold).astype(int)

        # Handle edge cases (all same class)
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            return cls(
                precision=0.0, recall=0.0, f1=0.0,
                auc_pr=float(n_pos > 0), auc_roc=0.5,
            )

        return cls(
            precision=float(precision_score(labels, preds, zero_division=0)),
            recall=float(recall_score(labels, preds, zero_division=0)),
            f1=float(f1_score(labels, preds, zero_division=0)),
            auc_pr=float(average_precision_score(labels, scores)),
            auc_roc=float(roc_auc_score(labels, scores)),
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc_pr": self.auc_pr,
            "auc_roc": self.auc_roc,
        }

    def __str__(self) -> str:
        return (
            f"AUC-PR={self.auc_pr:.4f} | AUC-ROC={self.auc_roc:.4f} | "
            f"P={self.precision:.4f} R={self.recall:.4f} F1={self.f1:.4f}"
        )
