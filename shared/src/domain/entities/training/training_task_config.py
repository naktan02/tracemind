"""TrainingTask의 objective/selection 설정 객체."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, TypeAlias

TrainingConfigScalar: TypeAlias = str | int | float | bool


@dataclass(slots=True)
class TrainingObjectiveConfig:
    """로컬 학습 objective 관련 설정."""

    loss: str = "diagonal_scale_heuristic"
    confidence_threshold: float | None = None
    margin_threshold: float | None = None
    extras: dict[str, TrainingConfigScalar] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, TrainingConfigScalar] | None,
    ) -> "TrainingObjectiveConfig":
        if source is None:
            return cls()
        return cls(
            loss=str(source.get("loss", "diagonal_scale_heuristic")),
            confidence_threshold=_optional_float(source.get("confidence_threshold")),
            margin_threshold=_optional_float(source.get("margin_threshold")),
            extras={
                key: value
                for key, value in source.items()
                if key not in {"loss", "confidence_threshold", "margin_threshold"}
            },
        )

    def to_mapping(self) -> dict[str, TrainingConfigScalar]:
        result: dict[str, TrainingConfigScalar] = {"loss": self.loss}
        if self.confidence_threshold is not None:
            result["confidence_threshold"] = self.confidence_threshold
        if self.margin_threshold is not None:
            result["margin_threshold"] = self.margin_threshold
        result.update(self.extras)
        return result


@dataclass(slots=True)
class TrainingSelectionPolicy:
    """로컬 학습 예시 선택 정책."""

    max_examples: int | None = None
    require_feedback: bool | None = None
    extras: dict[str, TrainingConfigScalar] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, TrainingConfigScalar] | None,
    ) -> "TrainingSelectionPolicy":
        if source is None:
            return cls()
        return cls(
            max_examples=_optional_int(source.get("max_examples")),
            require_feedback=_optional_bool(source.get("require_feedback")),
            extras={
                key: value
                for key, value in source.items()
                if key not in {"max_examples", "require_feedback"}
            },
        )

    def to_mapping(self) -> dict[str, TrainingConfigScalar]:
        result: dict[str, TrainingConfigScalar] = {}
        if self.max_examples is not None:
            result["max_examples"] = self.max_examples
        if self.require_feedback is not None:
            result["require_feedback"] = self.require_feedback
        result.update(self.extras)
        return result


def _optional_float(value: TrainingConfigScalar | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Expected float-like config value, got bool.")
    return float(value)


def _optional_int(value: TrainingConfigScalar | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Expected int-like config value, got bool.")
    return int(value)


def _optional_bool(value: TrainingConfigScalar | None) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("Expected bool config value.")
    return value
