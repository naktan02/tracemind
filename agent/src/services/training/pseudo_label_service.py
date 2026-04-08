"""Pseudo-label 후보 생성과 필터링 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from agent.src.services.training.acceptance_policies import (
    PseudoLabelAcceptancePolicy,
    Top1MarginThresholdAcceptancePolicy,
    build_pseudo_label_acceptance_policy,
)
from shared.src.config.training_defaults import (
    DEFAULT_TRAINING_PROFILE,
    TrainingDefaultsProfile,
)
from shared.src.contracts.training_contracts import (
    DecisionFeedbackSignal,
    TrainingTask,
)
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
)


@dataclass(slots=True)
class PseudoLabelSelectionResult:
    """pseudo-label 후보와 최종 채택 결과."""

    candidates: tuple[PseudoLabelCandidate, ...]
    accepted_candidates: tuple[PseudoLabelCandidate, ...]
    feedback_signals: tuple[DecisionFeedbackSignal, ...]

    @property
    def total_count(self) -> int:
        return len(self.candidates)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_candidates)

    @property
    def accepted_ratio(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.accepted_count / self.total_count


@dataclass(slots=True)
class PseudoLabelSelectionService:
    """score threshold와 margin 기준으로 pseudo-label을 선별한다."""

    default_profile: TrainingDefaultsProfile = field(
        default=DEFAULT_TRAINING_PROFILE
    )
    default_policy: PseudoLabelAcceptancePolicy = field(
        default_factory=Top1MarginThresholdAcceptancePolicy
    )

    def __post_init__(self) -> None:
        if self.default_acceptance_policy_name != self.default_policy.policy_name:
            raise ValueError(
                "Default pseudo-label acceptance policy does not match the "
                "configured default objective profile."
            )

    @property
    def default_confidence_threshold(self) -> float:
        return self.default_profile.confidence_threshold

    @property
    def default_margin_threshold(self) -> float:
        return self.default_profile.margin_threshold

    @property
    def default_acceptance_policy_name(self) -> str:
        return self.default_profile.acceptance_policy_name

    def select(
        self,
        *,
        scored_events: tuple[ScoredEvent, ...] | list[ScoredEvent],
        training_task: TrainingTask,
    ) -> PseudoLabelSelectionResult:
        confidence_threshold = (
            training_task.objective_config.confidence_threshold
            if training_task.objective_config.confidence_threshold is not None
            else self.default_confidence_threshold
        )
        margin_threshold = (
            training_task.objective_config.margin_threshold
            if training_task.objective_config.margin_threshold is not None
            else self.default_margin_threshold
        )
        acceptance_policy = self._resolve_policy(training_task=training_task)
        max_examples = training_task.selection_policy.max_examples

        initial_candidates = [
            self._build_candidate(
                scored_event=event,
                training_task=training_task,
                confidence_threshold=confidence_threshold,
                margin_threshold=margin_threshold,
                acceptance_policy=acceptance_policy,
            )
            for event in scored_events
        ]
        prelim_accepted = [
            candidate for candidate in initial_candidates if candidate.accepted
        ]
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

        finalized_candidates: list[PseudoLabelCandidate] = []
        accepted_candidates: list[PseudoLabelCandidate] = []
        feedback_signals: list[DecisionFeedbackSignal] = []
        for candidate in initial_candidates:
            threshold_accepted = candidate.accepted
            is_selected = candidate.candidate_id in selected_ids
            final_accepted = threshold_accepted and is_selected
            if final_accepted:
                selection_stage = "accepted"
            elif threshold_accepted:
                selection_stage = "dropped_by_cap"
            else:
                selection_stage = "threshold_rejected"

            metadata = dict(candidate.metadata)
            metadata["threshold_accepted"] = threshold_accepted
            metadata["selected_by_cap"] = is_selected
            metadata["final_accepted"] = final_accepted
            metadata["selection_stage"] = selection_stage
            metadata["max_examples"] = max_examples if max_examples is not None else -1
            if threshold_accepted:
                metadata["pre_cap_rank"] = pre_cap_ranks[candidate.candidate_id]

            finalized = replace(
                candidate,
                accepted=final_accepted,
                metadata=metadata,
            )
            finalized_candidates.append(finalized)
            if finalized.accepted:
                accepted_candidates.append(finalized)
                feedback_signals.append(
                    self._to_feedback_signal(
                        candidate=finalized,
                        training_task=training_task,
                    )
                )

        return PseudoLabelSelectionResult(
            candidates=tuple(finalized_candidates),
            accepted_candidates=tuple(accepted_candidates),
            feedback_signals=tuple(feedback_signals),
        )

    def _build_candidate(
        self,
        *,
        scored_event: ScoredEvent,
        training_task: TrainingTask,
        confidence_threshold: float,
        margin_threshold: float,
        acceptance_policy: PseudoLabelAcceptancePolicy,
    ) -> PseudoLabelCandidate:
        decision = acceptance_policy.evaluate(
            category_scores=scored_event.category_scores,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
        )

        return PseudoLabelCandidate(
            schema_version="pseudo_label_candidate.v1",
            candidate_id=f"{training_task.round_id}:{scored_event.query_id}",
            source_event_ref=scored_event.query_id,
            occurred_at=scored_event.occurred_at,
            label=decision.label,
            confidence=decision.confidence,
            margin=decision.margin,
            accepted=decision.accepted,
            runner_up_label=decision.runner_up_label,
            runner_up_score=decision.runner_up_score,
            task_id=training_task.task_id,
            round_id=training_task.round_id,
            metadata={
                "confidence_threshold": confidence_threshold,
                "margin_threshold": margin_threshold,
                "acceptance_policy_name": acceptance_policy.policy_name,
            },
        )

    def _resolve_policy(
        self,
        *,
        training_task: TrainingTask,
    ) -> PseudoLabelAcceptancePolicy:
        policy_name = (
            training_task.objective_config.acceptance_policy_name
            or self.default_acceptance_policy_name
        )
        if policy_name == self.default_policy.policy_name:
            return self.default_policy
        return build_pseudo_label_acceptance_policy(policy_name)

    @staticmethod
    def _to_feedback_signal(
        *,
        candidate: PseudoLabelCandidate,
        training_task: TrainingTask,
    ) -> DecisionFeedbackSignal:
        return DecisionFeedbackSignal(
            schema_version="decision_feedback_signal.v1",
            signal_id=f"signal:{candidate.candidate_id}",
            signal_type="pseudo_label",
            label=candidate.label,
            confidence=candidate.confidence,
            occurred_at=candidate.occurred_at,
            source_event_ref=candidate.source_event_ref,
            task_context={
                "task_id": training_task.task_id,
                "round_id": training_task.round_id,
                "margin": candidate.margin,
                "runner_up_score": candidate.runner_up_score or 0.0,
            },
        )
