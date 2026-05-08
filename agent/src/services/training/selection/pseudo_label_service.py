"""Pseudo-label 후보 생성과 필터링 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field

from methods.federated_ssl.runtime_fallbacks import (
    RUNTIME_FALLBACK_TRAINING_PROFILE,
    RuntimeFallbackTrainingProfile,
)
from methods.ssl.hooks.acceptance import PseudoLabelAcceptancePolicySpec
from methods.ssl.hooks.registry import (
    build_pseudo_label_acceptance_policy,
    build_pseudo_label_selection_hook,
)
from methods.ssl.hooks.selection import (
    MarginThresholdPseudoLabelSelectionHook,
    PseudoLabelSelectionConfig,
    PseudoLabelSelectionHook,
)
from shared.src.contracts.training_contracts import (
    DecisionFeedbackSignal,
    TrainingTask,
)
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
)
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

from .candidate_builder import PseudoLabelCandidateBuilder
from .evidence_service import (
    PseudoLabelEvidenceService,
)
from .selector import PseudoLabelSelector


def _build_default_acceptance_policy() -> PseudoLabelAcceptancePolicySpec:
    """runtime fallback의 acceptance policy를 methods-owned spec으로 해석한다."""

    return build_pseudo_label_acceptance_policy(
        RUNTIME_FALLBACK_TRAINING_PROFILE.acceptance_policy_name
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


@dataclass(slots=True)
class PseudoLabelSelectionService:
    """score threshold와 margin 기준으로 pseudo-label을 선별한다."""

    default_profile: RuntimeFallbackTrainingProfile = field(
        default=RUNTIME_FALLBACK_TRAINING_PROFILE
    )
    evidence_service: PseudoLabelEvidenceService = field(
        default_factory=PseudoLabelEvidenceService
    )
    default_policy: PseudoLabelAcceptancePolicySpec = field(
        default_factory=_build_default_acceptance_policy
    )
    default_selection_hook: PseudoLabelSelectionHook = field(
        default_factory=MarginThresholdPseudoLabelSelectionHook
    )
    selector: PseudoLabelSelector = field(default_factory=PseudoLabelSelector)

    def __post_init__(self) -> None:
        if self.default_acceptance_policy_name != self.default_policy.policy_name:
            raise ValueError(
                "Default pseudo-label acceptance policy does not match the "
                "configured default objective profile."
            )
        if (
            self.default_policy.selection_hook_name
            != self.default_pseudo_label_algorithm_name
        ):
            raise ValueError(
                "Default pseudo-label acceptance policy does not point to the "
                "configured default selection hook."
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
        candidate_builder = PseudoLabelCandidateBuilder(
            default_evidence_backend_name=self.default_profile.evidence_backend_name
        )

        built_candidates = tuple(
            candidate_builder.build(
                evidence=evidence,
                training_task=training_task,
                selection_config=selection_config,
                selection_hook=selection_hook,
            )
            for evidence in evidence_list
        )
        finalized = self.selector.finalize(
            built_candidates=built_candidates,
            training_task=training_task,
            selection_config=selection_config,
            max_examples=max_examples,
        )

        return PseudoLabelSelectionResult(
            evidences=evidence_list,
            candidates=finalized.candidates,
            accepted_candidates=finalized.accepted_candidates,
            feedback_signals=finalized.feedback_signals,
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
