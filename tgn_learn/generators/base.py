"""
Base class for synthetic fraud data generators.

All generators produce TemporalGraph instances with injected fraud patterns.
They are configurable, reproducible (via seeds), and produce labeled edges.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from tgn_learn.graph import TemporalGraph


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generation.

    Attributes:
        num_accounts: Number of account nodes to create
        num_merchants: Number of merchant nodes to create
        num_transactions: Total number of transactions (edges) to generate
        fraud_rate: Target fraction of fraudulent transactions (0.0 to 1.0)
        seed: Random seed for reproducibility (None = random)
        start_timestamp: Starting Unix timestamp for the simulation
        duration_days: Duration of the simulation in days
    """

    num_accounts: int = 500
    num_merchants: int = 50
    num_transactions: int = 5000
    fraud_rate: float = 0.02
    seed: Optional[int] = None
    start_timestamp: float = 1_700_000_000.0
    duration_days: int = 30


class BaseGenerator(ABC):
    """Abstract base class for fraud dataset generators.

    Subclasses must implement `generate()` to produce a TemporalGraph
    with injected fraud patterns.

    Example:
        >>> gen = BankSimGenerator(GeneratorConfig(seed=42))
        >>> graph = gen.generate()
        >>> print(graph.summary())
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this generator."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this generator produces."""
        ...

    @property
    @abstractmethod
    def fraud_patterns(self) -> list[str]:
        """List of fraud pattern names this generator can inject."""
        ...

    @abstractmethod
    def generate(self) -> TemporalGraph:
        """Generate a complete temporal graph with fraud patterns.

        Returns:
            A TemporalGraph with labeled edges (0=legit, 1=fraud)
        """
        ...
