"""로컬 pseudo-label 후보."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

PseudoLabelCandidateMetadataScalar = str | int | float | bool


class PseudoLabelSelectionStage(StrEnum):
    """Pseudo-label candidate 최종 selection 단계."""

    ACCEPTED = "accepted"
    DROPPED_BY_CAP = "dropped_by_cap"
    POLICY_REJECTED = "policy_rejected"


SELECTION_CONTEXT_COMPATIBILITY_METADATA_KEYS = frozenset(
    {
        "policy_accepted",
        "selected_by_cap",
        "final_accepted",
        "selection_stage",
        "pre_cap_rank",
        "max_examples",
        "pseudo_label_algorithm_name",
        "evidence_backend_name",
        "confidence_kind",
        "view_kind",
    }
)


@dataclass(frozen=True, slots=True)
class PseudoLabelSelectionContext:
    """Pseudo-label candidate의 typed selection semantics."""

    policy_accepted: bool
    selected_by_cap: bool
    final_accepted: bool
    selection_stage: PseudoLabelSelectionStage
    selection_parameters: dict[str, float] = field(default_factory=dict)
    max_examples: int | None = None
    pre_cap_rank: int | None = None
    pseudo_label_algorithm_name: str | None = None
    evidence_backend_name: str | None = None
    evidence_confidence_kind: str | None = None
    evidence_view_kind: str | None = None

    def __post_init__(self) -> None:
        if self.pre_cap_rank is not None and self.pre_cap_rank < 1:
            raise ValueError("pre_cap_rank must be >= 1 when provided.")
        if (
            self.selection_stage == PseudoLabelSelectionStage.ACCEPTED
            and not self.final_accepted
        ):
            raise ValueError("selection_stage='accepted' requires final_accepted=True.")

    def to_compatibility_metadata(
        self,
    ) -> dict[str, PseudoLabelCandidateMetadataScalar]:
        """기존 candidate.metadata shape와 맞추는 compatibility mapping."""

        metadata: dict[str, PseudoLabelCandidateMetadataScalar] = {
            "policy_accepted": self.policy_accepted,
            "selected_by_cap": self.selected_by_cap,
            "final_accepted": self.final_accepted,
            "selection_stage": self.selection_stage.value,
        }
        if self.pre_cap_rank is not None:
            metadata["pre_cap_rank"] = self.pre_cap_rank
        for key, value in self.selection_parameters.items():
            metadata[f"selection_parameter.{key}"] = value
        if self.max_examples is not None:
            metadata["max_examples"] = self.max_examples
        if self.pseudo_label_algorithm_name is not None:
            metadata["pseudo_label_algorithm_name"] = self.pseudo_label_algorithm_name
        if self.evidence_backend_name is not None:
            metadata["evidence_backend_name"] = self.evidence_backend_name
        if self.evidence_confidence_kind is not None:
            metadata["confidence_kind"] = self.evidence_confidence_kind
        if self.evidence_view_kind is not None:
            metadata["view_kind"] = self.evidence_view_kind
        return metadata


@dataclass(slots=True)
class PseudoLabelCandidate:
    """prototype score에서 뽑은 로컬 pseudo-label 후보."""

    schema_version: str
    candidate_id: str
    source_event_ref: str
    occurred_at: datetime
    label: str
    confidence: float
    margin: float
    accepted: bool
    runner_up_label: str | None = None
    runner_up_score: float | None = None
    evidence_ref: str | None = None
    confidence_kind: str | None = None
    sample_weight: float = 1.0
    task_id: str | None = None
    round_id: str | None = None
    selection_context: PseudoLabelSelectionContext | None = None
    metadata: dict[str, PseudoLabelCandidateMetadataScalar] = field(
        default_factory=dict
    )
