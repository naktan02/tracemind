"""Pseudo-label selection cap/ranking policy."""

from __future__ import annotations

from dataclasses import dataclass

from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
)


@dataclass(frozen=True, slots=True)
class PseudoLabelCapDecision:
    """Threshold 통과 후보에 대한 cap 적용 결과."""

    selected_candidate_ids: frozenset[str]
    pre_cap_ranks: dict[str, int]


@dataclass(frozen=True, slots=True)
class PseudoLabelCapPolicy:
    """Confidence/margin ranking 뒤 max_examples cap을 적용한다."""

    def decide(
        self,
        *,
        candidates: tuple[PseudoLabelCandidate, ...],
        max_examples: int | None,
    ) -> PseudoLabelCapDecision:
        prelim_accepted = [candidate for candidate in candidates if candidate.accepted]
        prelim_accepted.sort(
            key=lambda candidate: (
                -candidate.confidence,
                -candidate.margin,
                candidate.source_event_ref,
            )
        )

        if max_examples is not None:
            selected_ids = {
                candidate.candidate_id
                for candidate in prelim_accepted[: max(max_examples, 0)]
            }
        else:
            selected_ids = {candidate.candidate_id for candidate in prelim_accepted}
        pre_cap_ranks = {
            candidate.candidate_id: index + 1
            for index, candidate in enumerate(prelim_accepted)
        }
        return PseudoLabelCapDecision(
            selected_candidate_ids=frozenset(selected_ids),
            pre_cap_ranks=pre_cap_ranks,
        )
