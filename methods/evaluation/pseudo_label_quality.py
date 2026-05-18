"""Pseudo-label 선택 품질 diagnostic 계산."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class PseudoLabelCandidateLike(Protocol):
    """Pseudo-label quality summary가 요구하는 candidate surface."""

    source_event_ref: str
    label: str
    confidence: float
    margin: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class PseudoLabelQualitySummary:
    """client pseudo-label selection quality 요약."""

    pseudo_label_confidence_mean: float | None
    pseudo_label_margin_mean: float | None
    pseudo_label_correct_count: int
    pseudo_label_evaluated_count: int
    accepted_label_distribution: dict[str, int]
    rejected_label_distribution: dict[str, int]

    @classmethod
    def empty(cls) -> "PseudoLabelQualitySummary":
        return cls(
            pseudo_label_confidence_mean=None,
            pseudo_label_margin_mean=None,
            pseudo_label_correct_count=0,
            pseudo_label_evaluated_count=0,
            accepted_label_distribution={},
            rejected_label_distribution={},
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "pseudo_label_confidence_mean": self.pseudo_label_confidence_mean,
            "pseudo_label_margin_mean": self.pseudo_label_margin_mean,
            "pseudo_label_correct_count": self.pseudo_label_correct_count,
            "pseudo_label_evaluated_count": self.pseudo_label_evaluated_count,
            "accepted_label_distribution": dict(self.accepted_label_distribution),
            "rejected_label_distribution": dict(self.rejected_label_distribution),
        }


@dataclass(frozen=True, slots=True)
class PseudoLabelCandidateRecord:
    """최종 classifier snapshot에서 만든 pseudo-label candidate record."""

    source_event_ref: str
    label: str
    confidence: float
    margin: float
    accepted: bool


def build_pseudo_label_quality_summary(
    *,
    candidates: Sequence[PseudoLabelCandidateLike],
    rows_with_simulation_labels: Sequence[Mapping[str, object]],
) -> PseudoLabelQualitySummary:
    """simulation label을 아는 경우 accepted pseudo-label 품질을 요약한다."""

    accepted_candidates = tuple(
        candidate for candidate in candidates if candidate.accepted
    )
    rejected_candidates = tuple(
        candidate for candidate in candidates if not candidate.accepted
    )
    true_label_by_query_id = {
        str(row["query_id"]): str(row["mapped_label_4"])
        for row in rows_with_simulation_labels
        if "query_id" in row and "mapped_label_4" in row
    }
    evaluated_candidates = tuple(
        candidate
        for candidate in accepted_candidates
        if str(candidate.source_event_ref) in true_label_by_query_id
    )
    correct_count = sum(
        1
        for candidate in evaluated_candidates
        if true_label_by_query_id[str(candidate.source_event_ref)]
        == str(candidate.label)
    )
    return PseudoLabelQualitySummary(
        pseudo_label_confidence_mean=_mean_candidate_value(
            tuple(candidates),
            "confidence",
        ),
        pseudo_label_margin_mean=_mean_candidate_value(tuple(candidates), "margin"),
        pseudo_label_correct_count=correct_count,
        pseudo_label_evaluated_count=len(evaluated_candidates),
        accepted_label_distribution=_candidate_label_distribution(accepted_candidates),
        rejected_label_distribution=_candidate_label_distribution(rejected_candidates),
    )


def _mean_candidate_value(
    candidates: tuple[PseudoLabelCandidateLike, ...],
    field_name: str,
) -> float | None:
    values = [float(getattr(candidate, field_name)) for candidate in candidates]
    if not values:
        return None
    return sum(values) / len(values)


def _candidate_label_distribution(
    candidates: tuple[PseudoLabelCandidateLike, ...],
) -> dict[str, int]:
    return dict(
        sorted(Counter(str(candidate.label) for candidate in candidates).items())
    )
