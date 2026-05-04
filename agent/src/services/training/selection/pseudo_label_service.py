"""Pseudo-label 후보 생성과 필터링 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from agent.src.services.training.acceptance_policies.base import (
    PseudoLabelAcceptancePolicy,
)
from agent.src.services.training.acceptance_policies.top1 import (
    Top1MarginThresholdAcceptancePolicy,
)
from methods.ssl.pseudo_label_selection.base import (
    PseudoLabelSelectionConfig,
    PseudoLabelSelectionHook,
)
from methods.ssl.pseudo_label_selection.margin_threshold import (
    MarginThresholdPseudoLabelSelectionMethod,
)
from methods.ssl.pseudo_label_selection.registry import (
    build_pseudo_label_selection_hook,
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
    PseudoLabelSelectionContext,
    PseudoLabelSelectionStage,
)
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

from .evidence_service import (
    PseudoLabelEvidenceService,
)


@dataclass(slots=True)
class PseudoLabelSelectionResult:
    """pseudo-label 후보와 최종 채택 결과."""

    candidates: tuple[PseudoLabelCandidate, ...]
    accepted_candidates: tuple[PseudoLabelCandidate, ...]
    feedback_signals: tuple[DecisionFeedbackSignal, ...]
    evidences: tuple[PseudoLabelEvidence, ...] = ()

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


@dataclass(frozen=True, slots=True)
class _SelectionContextSeed:
    pseudo_label_algorithm_name: str
    evidence_backend_name: str
    evidence_confidence_kind: str
    evidence_view_kind: str


@dataclass(frozen=True, slots=True)
class _BuiltCandidate:
    candidate: PseudoLabelCandidate
    context_seed: _SelectionContextSeed


@dataclass(slots=True)
class PseudoLabelSelectionService:
    """score threshold와 margin 기준으로 pseudo-label을 선별한다."""

    default_profile: TrainingDefaultsProfile = field(default=DEFAULT_TRAINING_PROFILE)
    evidence_service: PseudoLabelEvidenceService = field(
        default_factory=PseudoLabelEvidenceService
    )
    default_policy: PseudoLabelAcceptancePolicy = field(
        default_factory=Top1MarginThresholdAcceptancePolicy
    )
    default_selection_hook: PseudoLabelSelectionHook = field(
        default_factory=MarginThresholdPseudoLabelSelectionMethod
    )

    def __post_init__(self) -> None:
        if self.default_acceptance_policy_name != self.default_policy.policy_name:
            raise ValueError(
                "Default pseudo-label acceptance policy does not match the "
                "configured default objective profile."
            )
        if (
            self.default_pseudo_label_algorithm_name
            != self.default_selection_hook.hook_name
        ):
            raise ValueError(
                "Default pseudo-label selection hook does not match the configured "
                "default objective profile."
            )
        if (
            self.default_profile.evidence_backend_name
            != self.evidence_service.default_evidence_backend_name
        ):
            raise ValueError(
                "Default pseudo-label evidence backend does not match the "
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

    @property
    def default_pseudo_label_algorithm_name(self) -> str:
        return self.default_profile.pseudo_label_algorithm_name

    def select(
        self,
        *,
        scored_events: tuple[ScoredEvent, ...] | list[ScoredEvent],
        training_task: TrainingTask,
    ) -> PseudoLabelSelectionResult:
        evidences = self.evidence_service.build_evidences(
            scored_events=scored_events,
            training_task=training_task,
        )
        return self.select_evidences(
            evidences=evidences,
            training_task=training_task,
        )

    def select_evidences(
        self,
        *,
        evidences: tuple[PseudoLabelEvidence, ...] | list[PseudoLabelEvidence],
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
        selection_hook = self._resolve_selection_hook(training_task=training_task)
        max_examples = training_task.selection_policy.max_examples
        evidence_list = tuple(evidences)
        selection_config = PseudoLabelSelectionConfig(
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
        )

        built_candidates = [
            self._build_candidate(
                evidence=evidence,
                training_task=training_task,
                selection_config=selection_config,
                selection_hook=selection_hook,
            )
            for evidence in evidence_list
        ]
        initial_candidates = tuple(
            built_candidate.candidate for built_candidate in built_candidates
        )
        context_seed_by_candidate_id = {
            built_candidate.candidate.candidate_id: built_candidate.context_seed
            for built_candidate in built_candidates
        }
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

            selection_context = self._build_selection_context(
                threshold_accepted=threshold_accepted,
                selected_by_cap=is_selected,
                final_accepted=final_accepted,
                selection_stage=selection_stage,
                context_seed=context_seed_by_candidate_id[candidate.candidate_id],
                pre_cap_rank=(
                    None
                    if not threshold_accepted
                    else pre_cap_ranks[candidate.candidate_id]
                ),
                confidence_threshold=selection_config.confidence_threshold,
                margin_threshold=selection_config.margin_threshold,
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
                    self._to_feedback_signal(
                        candidate=finalized,
                        training_task=training_task,
                    )
                )

        return PseudoLabelSelectionResult(
            evidences=evidence_list,
            candidates=tuple(finalized_candidates),
            accepted_candidates=tuple(accepted_candidates),
            feedback_signals=tuple(feedback_signals),
        )

    def _build_candidate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        training_task: TrainingTask,
        selection_config: PseudoLabelSelectionConfig,
        selection_hook: PseudoLabelSelectionHook,
    ) -> _BuiltCandidate:
        decision = selection_hook.evaluate(
            evidence=evidence,
            config=selection_config,
        )

        return _BuiltCandidate(
            candidate=PseudoLabelCandidate(
                schema_version="pseudo_label_candidate.v1",
                candidate_id=f"{training_task.round_id}:{evidence.source_event_ref}",
                source_event_ref=evidence.source_event_ref,
                occurred_at=evidence.occurred_at,
                label=decision.label,
                confidence=decision.confidence,
                margin=decision.margin,
                accepted=decision.accepted,
                runner_up_label=decision.runner_up_label,
                runner_up_score=decision.runner_up_score,
                evidence_ref=evidence.evidence_id,
                confidence_kind=decision.confidence_kind,
                sample_weight=decision.sample_weight,
                task_id=training_task.task_id,
                round_id=training_task.round_id,
            ),
            context_seed=_SelectionContextSeed(
                pseudo_label_algorithm_name=selection_hook.hook_name,
                evidence_backend_name=self._resolve_evidence_backend_name(
                    evidence=evidence,
                    training_task=training_task,
                ),
                evidence_confidence_kind=evidence.confidence_kind,
                evidence_view_kind=evidence.view_kind,
            ),
        )

    def _build_selection_context(
        self,
        *,
        threshold_accepted: bool,
        selected_by_cap: bool,
        final_accepted: bool,
        selection_stage: str,
        context_seed: _SelectionContextSeed,
        pre_cap_rank: int | None,
        confidence_threshold: float,
        margin_threshold: float,
        max_examples: int | None,
    ) -> PseudoLabelSelectionContext:
        return PseudoLabelSelectionContext(
            threshold_accepted=threshold_accepted,
            selected_by_cap=selected_by_cap,
            final_accepted=final_accepted,
            selection_stage=PseudoLabelSelectionStage(selection_stage),
            pre_cap_rank=pre_cap_rank,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            max_examples=max_examples,
            pseudo_label_algorithm_name=context_seed.pseudo_label_algorithm_name,
            evidence_backend_name=context_seed.evidence_backend_name,
            evidence_confidence_kind=context_seed.evidence_confidence_kind,
            evidence_view_kind=context_seed.evidence_view_kind,
        )

    def _resolve_selection_hook(
        self,
        *,
        training_task: TrainingTask,
    ) -> PseudoLabelSelectionHook:
        algorithm_name = (
            training_task.objective_config.pseudo_label_algorithm_name
            or self.default_pseudo_label_algorithm_name
        )
        if algorithm_name == self.default_selection_hook.hook_name:
            return self.default_selection_hook
        return build_pseudo_label_selection_hook(algorithm_name)

    def _resolve_evidence_backend_name(
        self,
        *,
        evidence: PseudoLabelEvidence,
        training_task: TrainingTask,
    ) -> str:
        evidence_backend_name = evidence.metadata.get("evidence_backend_name")
        if isinstance(evidence_backend_name, str) and evidence_backend_name.strip():
            return evidence_backend_name
        return (
            training_task.objective_config.evidence_backend_name
            or self.default_profile.evidence_backend_name
        )

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
                "confidence_kind": candidate.confidence_kind or "unknown",
                "sample_weight": candidate.sample_weight,
                "evidence_ref": candidate.evidence_ref or "",
            },
        )
