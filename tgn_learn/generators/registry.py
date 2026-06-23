"""
Generator Registry — discover and instantiate available generators.

Provides a central catalog of all available synthetic data generators,
making it easy to list options and create instances programmatically.
"""

from __future__ import annotations

from typing import Type

from .base import BaseGenerator, GeneratorConfig
from .banksim import BankSimGenerator
from .paysim import PaySimGenerator


class GeneratorRegistry:
    """
    Registry of available synthetic data generators.

    Example:
        >>> registry = GeneratorRegistry()
        >>> print(registry.list_generators())
        >>> gen = registry.create("banksim", GeneratorConfig(seed=42))
        >>> graph = gen.generate()
    """

    _generators: dict[str, Type[BaseGenerator]] = {
        "banksim": BankSimGenerator,
        "paysim": PaySimGenerator,
    }

    @classmethod
    def list_generators(cls) -> list[dict[str, str]]:
        """List all available generators with metadata."""
        result = []
        for name, gen_cls in cls._generators.items():
            result.append({
                "name": name,
                "description": gen_cls.description,
                "fraud_patterns": gen_cls.fraud_patterns,
            })
        return result

    @classmethod
    def create(cls, name: str, config: GeneratorConfig, **kwargs) -> BaseGenerator:
        """Create a generator instance by name.

        Args:
            name: Generator name (e.g. 'banksim', 'paysim')
            config: Generator configuration
            **kwargs: Additional generator-specific arguments

        Returns:
            Configured generator instance

        Raises:
            KeyError: If generator name is not found
        """
        if name not in cls._generators:
            available = ", ".join(cls._generators.keys())
            raise KeyError(f"Unknown generator '{name}'. Available: {available}")
        return cls._generators[name](config, **kwargs)

    @classmethod
    def register(cls, gen_cls: Type[BaseGenerator]) -> None:
        """Register a new generator class."""
        cls._generators[gen_cls.name] = gen_cls
