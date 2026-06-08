"""Typed pseudo-label selection context construction."""

from __future__ import annotations

from dataclasses import dataclass

from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelSelectionContext,
    PseudoLabelSelectionStage,
)

from .candidate_builder import SelectionContextSeed


@dataclass(frozen=True, slots=True)
class PseudoLabelSelectionContextBuilder:
    """Selection decision metadata를 typed context로 고정한다."""

    def build(
        self,
        *,
        policy_accepted: bool,
        selected_by_cap: bool,
        final_accepted: bool,
        selection_stage: PseudoLabelSelectionStage,
        context_seed: SelectionContextSeed,
        pre_cap_rank: int | None,
        selection_parameters: dict[str, float],
        max_examples: int | None,
    ) -> PseudoLabelSelectionContext:
        return PseudoLabelSelectionContext(
            policy_accepted=policy_accepted,
            selected_by_cap=selected_by_cap,
            final_accepted=final_accepted,
            selection_stage=selection_stage,
            pre_cap_rank=pre_cap_rank,
            selection_parameters=selection_parameters,
            max_examples=max_examples,
            pseudo_label_algorithm_name=context_seed.pseudo_label_algorithm_name,
            evidence_backend_name=context_seed.evidence_backend_name,
            evidence_view_kind=context_seed.evidence_view_kind,
        )
