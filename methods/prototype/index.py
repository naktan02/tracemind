"""Prototype index value objects for analysis and method adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PrototypeVector:
    """One cluster/centroid prototype vector."""

    prototype_id: str
    centroid: list[float]
    member_count: int


@dataclass(slots=True)
class PrototypeIndex:
    """Strategy-specific category to prototype vectors index."""

    strategy_name: str
    categories: dict[str, list[PrototypeVector]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prototype_count(self) -> int:
        return sum(len(prototypes) for prototypes in self.categories.values())

    def prototype_count_by_category(self) -> dict[str, int]:
        return {
            category: len(prototypes)
            for category, prototypes in sorted(self.categories.items())
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "prototype_count": self.prototype_count,
            "prototype_count_by_category": self.prototype_count_by_category(),
            "metadata": self.metadata,
            "categories": {
                category: [asdict(prototype) for prototype in prototypes]
                for category, prototypes in sorted(self.categories.items())
            },
        }
