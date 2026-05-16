"""Diagonal scale family의 공용 local training backend 설정값."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from methods.common.config_reading import (
    ConfigScalar,
    freeze_mapping,
    read_float,
    validate_allowed_keys,
)

DiagonalScaleConfigScalar = ConfigScalar
TRAINING_BACKEND_EXTRA_SCOPE = "training_backend"
DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_EXTRA_KEYS = (
    "delta_scale_multiplier",
    "max_abs_delta",
    "minimum_effective_scale",
)


@dataclass(frozen=True, slots=True)
class DiagonalScaleHeuristicTrainingBackendConfig:
    """Diagonal-scale heuristic local training backend 설정."""

    delta_scale_multiplier: float = 10.0
    max_abs_delta: float = 0.05
    minimum_effective_scale: float = 1e-4

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, DiagonalScaleConfigScalar] | None,
    ) -> "DiagonalScaleHeuristicTrainingBackendConfig":
        validate_allowed_keys(
            source,
            allowed_keys=DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_EXTRA_KEYS,
            config_name="diagonal-scale heuristic training backend config",
        )
        return cls(
            delta_scale_multiplier=read_float(
                source,
                "delta_scale_multiplier",
                DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG.delta_scale_multiplier,
            ),
            max_abs_delta=read_float(
                source,
                "max_abs_delta",
                DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG.max_abs_delta,
            ),
            minimum_effective_scale=read_float(
                source,
                "minimum_effective_scale",
                DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG.minimum_effective_scale,
            ),
        )

    def to_mapping(self) -> Mapping[str, DiagonalScaleConfigScalar]:
        return freeze_mapping(
            {
                "delta_scale_multiplier": self.delta_scale_multiplier,
                "max_abs_delta": self.max_abs_delta,
                "minimum_effective_scale": self.minimum_effective_scale,
            }
        )

    def to_scoped_mapping(
        self,
        *,
        scope: str = TRAINING_BACKEND_EXTRA_SCOPE,
    ) -> Mapping[str, DiagonalScaleConfigScalar]:
        return freeze_mapping(
            {
                f"{scope}.delta_scale_multiplier": self.delta_scale_multiplier,
                f"{scope}.max_abs_delta": self.max_abs_delta,
                f"{scope}.minimum_effective_scale": self.minimum_effective_scale,
            }
        )


DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG = (
    DiagonalScaleHeuristicTrainingBackendConfig()
)
