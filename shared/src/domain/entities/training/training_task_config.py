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
    score_policy_name: str | None = None
    score_top_k: int | None = None
    acceptance_policy_name: str | None = None
    privacy_guard_name: str | None = None
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
            score_policy_name=_optional_str(source.get("score_policy_name")),
            score_top_k=_optional_positive_int(source.get("score_top_k")),
            acceptance_policy_name=_optional_str(
                source.get("acceptance_policy_name")
            ),
            privacy_guard_name=_optional_str(source.get("privacy_guard_name")),
            extras={
                key: value
                for key, value in source.items()
                if key
                not in {
                    "loss",
                    "confidence_threshold",
                    "margin_threshold",
                    "score_policy_name",
                    "score_top_k",
                    "acceptance_policy_name",
                    "privacy_guard_name",
                }
            },
        )

    def to_mapping(self) -> dict[str, TrainingConfigScalar]:
        result: dict[str, TrainingConfigScalar] = {"loss": self.loss}
        if self.confidence_threshold is not None:
            result["confidence_threshold"] = self.confidence_threshold
        if self.margin_threshold is not None:
            result["margin_threshold"] = self.margin_threshold
        if self.score_policy_name is not None:
            result["score_policy_name"] = self.score_policy_name
        if self.score_top_k is not None:
            result["score_top_k"] = self.score_top_k
        if self.acceptance_policy_name is not None:
            result["acceptance_policy_name"] = self.acceptance_policy_name
        if self.privacy_guard_name is not None:
            result["privacy_guard_name"] = self.privacy_guard_name
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


def _optional_str(value: TrainingConfigScalar | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_positive_int(value: TrainingConfigScalar | None) -> int | None:
    parsed = _optional_int(value)
    if parsed is None:
        return None
    if parsed < 1:
        raise ValueError("Expected positive int config value.")
    return parsed
