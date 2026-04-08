"""Diagonal scale family의 공용 local training backend 설정값."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

DiagonalScaleConfigScalar = str | int | float | bool
TRAINING_BACKEND_EXTRA_SCOPE = "training_backend"
DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_EXTRA_KEYS = (
    "delta_scale_multiplier",
    "max_abs_delta",
    "minimum_effective_scale",
)


def _freeze_mapping(
    source: Mapping[str, DiagonalScaleConfigScalar],
) -> Mapping[str, DiagonalScaleConfigScalar]:
    return MappingProxyType(dict(source))


def _read_float(
    source: Mapping[str, DiagonalScaleConfigScalar] | None,
    key: str,
    default: float,
) -> float:
    if source is None:
        return default
    value = source.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must not be bool.")
    return float(value)


def _validate_allowed_keys(
    source: Mapping[str, DiagonalScaleConfigScalar] | None,
    *,
    allowed_keys: tuple[str, ...],
    config_name: str,
) -> None:
    if source is None:
        return
    unexpected_keys = sorted(key for key in source if key not in allowed_keys)
    if unexpected_keys:
        raise ValueError(
            f"Unsupported {config_name} key(s): {unexpected_keys}."
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
        _validate_allowed_keys(
            source,
            allowed_keys=DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_EXTRA_KEYS,
            config_name="diagonal-scale heuristic training backend config",
        )
        return cls(
            delta_scale_multiplier=_read_float(
                source,
                "delta_scale_multiplier",
                DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG.delta_scale_multiplier,
            ),
            max_abs_delta=_read_float(
                source,
                "max_abs_delta",
                DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG.max_abs_delta,
            ),
            minimum_effective_scale=_read_float(
                source,
                "minimum_effective_scale",
                DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG.minimum_effective_scale,
            ),
        )

    def to_mapping(self) -> Mapping[str, DiagonalScaleConfigScalar]:
        return _freeze_mapping(
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
        return _freeze_mapping(
            {
                f"{scope}.delta_scale_multiplier": self.delta_scale_multiplier,
                f"{scope}.max_abs_delta": self.max_abs_delta,
                f"{scope}.minimum_effective_scale": self.minimum_effective_scale,
            }
        )


DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG = (
    DiagonalScaleHeuristicTrainingBackendConfig()
)


__all__ = [
    "DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG",
    "DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_EXTRA_KEYS",
    "DiagonalScaleConfigScalar",
    "DiagonalScaleHeuristicTrainingBackendConfig",
    "TRAINING_BACKEND_EXTRA_SCOPE",
]
