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
    PseudoLabelSelectionConfig,
    PseudoLabelSelectionHook,
    Top1RankedPseudoLabelSelectionHook,
)
from shared.src.contracts.training_contracts import (
    DecisionFeedbackSignal,
    TrainingConfigScalar,
    TrainingTask,
)
from shared.src.domain.entities.inference.events import AnalysisEvent
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

_SELECTION_PARAMETER_SCOPE = "selection"


def _build_default_acceptance_policy() -> PseudoLabelAcceptancePolicySpec:
    """runtime fallback의 acceptance policy를 methods-owned spec으로 해석한다."""

    return build_pseudo_label_acceptance_policy(
        RUNTIME_FALLBACK_TRAINING_PROFILE.acceptance_policy_name
    )


def _selection_parameters(training_task: TrainingTask) -> dict[str, float]:
    """method-owned selection parameter extras를 숫자 mapping으로 정규화한다."""

    raw_parameters = training_task.objective_config.get_component_extras(
        _SELECTION_PARAMETER_SCOPE,
        legacy_keys=("confidence_threshold", "margin_threshold"),
    )
    return {
        key: _coerce_float_parameter(key, value)
        for key, value in raw_parameters.items()
    }


def _coerce_float_parameter(key: str, value: TrainingConfigScalar) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Selection parameter must not be bool: {key}")
    return float(value)


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
    """method-owned selection hook으로 pseudo-label을 선별한다."""

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
        default_factory=Top1RankedPseudoLabelSelectionHook
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
    def default_acceptance_policy_name(self) -> str:
        return self.default_profile.acceptance_policy_name

    @property
    def default_pseudo_label_algorithm_name(self) -> str:
        return self.default_profile.pseudo_label_algorithm_name

    def select(
        self,
        *,
        analysis_events: tuple[AnalysisEvent, ...] | list[AnalysisEvent],
        training_task: TrainingTask,
    ) -> PseudoLabelSelectionResult:
        evidences = self.evidence_service.build_evidences(
            analysis_events=analysis_events,
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
        selection_hook = self._resolve_selection_hook(training_task=training_task)
        max_examples = training_task.selection_policy.max_examples
        evidence_list = tuple(evidences)
        selection_config = PseudoLabelSelectionConfig(
            parameters=_selection_parameters(training_task),
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
