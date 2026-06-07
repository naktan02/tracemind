"""Pseudo-label selection finalization orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from methods.ssl.hooks.selection import PseudoLabelSelectionConfig
from shared.src.contracts.training_contracts import (
    DecisionFeedbackSignal,
    TrainingTask,
)
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
    PseudoLabelSelectionStage,
)

from .candidate_builder import BuiltPseudoLabelCandidate
from .cap_policy import PseudoLabelCapPolicy
from .feedback_builder import PseudoLabelFeedbackBuilder
from .selection_context_builder import PseudoLabelSelectionContextBuilder


@dataclass(frozen=True, slots=True)
class FinalizedPseudoLabelSelection:
    """Cap 적용 뒤 최종 candidate와 feedback 묶음."""

    candidates: tuple[PseudoLabelCandidate, ...]
    accepted_candidates: tuple[PseudoLabelCandidate, ...]
    feedback_signals: tuple[DecisionFeedbackSignal, ...]


@dataclass(slots=True)
class PseudoLabelSelector:
    """Threshold 후보를 최종 accepted/rejected candidate로 확정한다."""

    cap_policy: PseudoLabelCapPolicy = field(default_factory=PseudoLabelCapPolicy)
    context_builder: PseudoLabelSelectionContextBuilder = field(
        default_factory=PseudoLabelSelectionContextBuilder
    )
    feedback_builder: PseudoLabelFeedbackBuilder = field(
        default_factory=PseudoLabelFeedbackBuilder
    )

    def finalize(
        self,
        *,
        built_candidates: tuple[BuiltPseudoLabelCandidate, ...],
        training_task: TrainingTask,
        selection_config: PseudoLabelSelectionConfig,
        max_examples: int | None,
    ) -> FinalizedPseudoLabelSelection:
        initial_candidates = tuple(
            built_candidate.candidate for built_candidate in built_candidates
        )
        context_seed_by_candidate_id = {
            built_candidate.candidate.candidate_id: built_candidate.context_seed
            for built_candidate in built_candidates
        }
        cap_decision = self.cap_policy.decide(
            candidates=initial_candidates,
            max_examples=max_examples,
        )

        finalized_candidates: list[PseudoLabelCandidate] = []
        accepted_candidates: list[PseudoLabelCandidate] = []
        feedback_signals: list[DecisionFeedbackSignal] = []
        for candidate in initial_candidates:
            policy_accepted = candidate.accepted
            is_selected = candidate.candidate_id in (
                cap_decision.selected_candidate_ids
            )
            final_accepted = policy_accepted and is_selected
            if final_accepted:
                selection_stage = PseudoLabelSelectionStage.ACCEPTED
            elif policy_accepted:
                selection_stage = PseudoLabelSelectionStage.DROPPED_BY_CAP
            else:
                selection_stage = PseudoLabelSelectionStage.POLICY_REJECTED

            selection_context = self.context_builder.build(
                policy_accepted=policy_accepted,
                selected_by_cap=is_selected,
                final_accepted=final_accepted,
                selection_stage=selection_stage,
                context_seed=context_seed_by_candidate_id[candidate.candidate_id],
                pre_cap_rank=(
                    None
                    if not policy_accepted
                    else cap_decision.pre_cap_ranks[candidate.candidate_id]
                ),
                selection_parameters=dict(selection_config.parameters),
                max_examples=max_examples,
            )

            finalized = replace(
                candidate,
                accepted=final_accepted,
                selection_context=selection_context,
                metadata=selection_context.to_compatibility_metadata(),
            )
            finalized_candidates.append(finalized)
            if finalized.accepted:
                accepted_candidates.append(finalized)
                feedback_signals.append(
                    self.feedback_builder.build(
                        candidate=finalized,
                        training_task=training_task,
                    )
                )

        return FinalizedPseudoLabelSelection(
            candidates=tuple(finalized_candidates),
            accepted_candidates=tuple(accepted_candidates),
            feedback_signals=tuple(feedback_signals),
        )
