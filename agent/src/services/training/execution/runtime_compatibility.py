"""로컬 학습 runtime 조합 호환성 검증."""

from __future__ import annotations

from dataclasses import dataclass

from agent.src.services.inference.scoring_backends.registry import build_scoring_backend
from agent.src.services.training.backends.evidence.resolver import (
    resolve_pseudo_label_evidence_backend,
)
from agent.src.services.training.backends.inputs.resolver import (
    resolve_training_example_backend,
)
from agent.src.services.training.execution.privacy_guards.base import (
    SharedAdapterPrivacyGuard,
)
from agent.src.services.training.execution.privacy_guards.registry import (
    build_shared_adapter_privacy_guard,
)
from methods.adaptation.local_update_backend import (
    SharedAdapterTrainingBackend,
)
from methods.adaptation.local_update_registry import (
    build_shared_adapter_training_backend,
)
from methods.federated_ssl.runtime_fallbacks import RUNTIME_FALLBACK_TRAINING_PROFILE
from methods.ssl.hooks.acceptance import PseudoLabelAcceptancePolicySpec
from methods.ssl.hooks.registry import build_pseudo_label_acceptance_policy
from shared.src.contracts.training_contracts import TrainingTask

ANY_ADAPTER_KIND = "*"


@dataclass(slots=True)
class LocalTrainingRuntimeCompatibility:
    """검증된 로컬 학습 runtime 조합 요약."""

    adapter_kind: str
    training_backend_name: str
    example_generation_backend_name: str
    evidence_backend_name: str
    scorer_backend_name: str
    acceptance_policy_name: str
    privacy_guard_name: str


def validate_live_agent_stored_event_runtime(
    training_task: TrainingTask,
    *,
    similarity_name: str = "cosine",
) -> LocalTrainingRuntimeCompatibility:
    """stored-event 기반 live agent 경로에서 허용되는 조합만 통과시킨다."""

    objective = training_task.objective_config
    resolved_training_backend = build_shared_adapter_training_backend(
        objective.training_backend_name,
        objective_config=objective,
    )
    training_example_backend = resolve_training_example_backend(
        objective_config=objective,
        training_backend=resolved_training_backend,
    )
    scorer_backend_name = (
        objective.scorer_backend_name
        or RUNTIME_FALLBACK_TRAINING_PROFILE.scorer_backend_name
    )
    build_scoring_backend(
        scorer_backend_name,
        objective_config=objective,
        similarity_name=similarity_name,
    )
    compatibility = validate_local_training_runtime(
        training_task,
        similarity_name=similarity_name,
        training_backend=resolved_training_backend,
    )

    unsupported_reasons: list[str] = []
    if not training_example_backend.supports_stored_event_rebuild:
        unsupported_reasons.append(
            "stored-event 재구성을 지원하지 않는 example backend="
            f"{training_example_backend.backend_name}"
        )
    if unsupported_reasons:
        raise ValueError(
            "run-current-task does not support this runtime yet: "
            + "; ".join(unsupported_reasons)
        )
    return compatibility


def validate_local_training_runtime(
    training_task: TrainingTask,
    *,
    similarity_name: str = "cosine",
    default_acceptance_policy_name: str = (
        RUNTIME_FALLBACK_TRAINING_PROFILE.acceptance_policy_name
    ),
    default_privacy_guard_name: str = "noop",
    training_backend: SharedAdapterTrainingBackend | None = None,
    acceptance_policy: PseudoLabelAcceptancePolicySpec | None = None,
    privacy_guard: SharedAdapterPrivacyGuard | None = None,
) -> LocalTrainingRuntimeCompatibility:
    """TrainingTask가 선택한 로컬 runtime 조합이 서로 호환되는지 검증한다."""

    objective = training_task.objective_config
    resolved_training_backend = training_backend or (
        build_shared_adapter_training_backend(
            objective.training_backend_name,
            objective_config=objective,
        )
    )
    training_example_backend = resolve_training_example_backend(
        objective_config=objective,
        training_backend=resolved_training_backend,
    )
    evidence_backend = resolve_pseudo_label_evidence_backend(
        objective_config=objective,
    )
    scorer_backend_name = (
        objective.scorer_backend_name
        or RUNTIME_FALLBACK_TRAINING_PROFILE.scorer_backend_name
    )
    scorer_backend = build_scoring_backend(
        scorer_backend_name,
        objective_config=objective,
        similarity_name=similarity_name,
    )
    acceptance_policy_name = (
        objective.acceptance_policy_name or default_acceptance_policy_name
    )
    resolved_acceptance_policy = acceptance_policy or (
        build_pseudo_label_acceptance_policy(acceptance_policy_name)
    )
    pseudo_label_algorithm_name = (
        objective.pseudo_label_algorithm_name
        or RUNTIME_FALLBACK_TRAINING_PROFILE.pseudo_label_algorithm_name
    )
    if resolved_acceptance_policy.selection_hook_name != pseudo_label_algorithm_name:
        raise ValueError(
            "Incompatible acceptance policy: "
            f"{resolved_acceptance_policy.policy_name} maps to "
            f"{resolved_acceptance_policy.selection_hook_name}, but objective uses "
            f"pseudo_label_algorithm_name={pseudo_label_algorithm_name}."
        )
    privacy_guard_name = objective.privacy_guard_name or default_privacy_guard_name
    resolved_privacy_guard = privacy_guard or build_shared_adapter_privacy_guard(
        privacy_guard_name
    )

    _require_adapter_kind_support(
        component_type="evidence backend",
        component_name=evidence_backend.backend_name,
        supported_adapter_kinds=evidence_backend.supported_adapter_kinds,
        adapter_kind=resolved_training_backend.adapter_kind,
    )
    _require_adapter_kind_support(
        component_type="scoring backend",
        component_name=scorer_backend.backend_name,
        supported_adapter_kinds=scorer_backend.supported_adapter_kinds,
        adapter_kind=resolved_training_backend.adapter_kind,
    )
    _require_adapter_kind_support(
        component_type="acceptance policy",
        component_name=resolved_acceptance_policy.policy_name,
        supported_adapter_kinds=resolved_acceptance_policy.supported_adapter_kinds,
        adapter_kind=resolved_training_backend.adapter_kind,
    )
    _require_adapter_kind_support(
        component_type="privacy guard",
        component_name=resolved_privacy_guard.guard_name,
        supported_adapter_kinds=resolved_privacy_guard.supported_adapter_kinds,
        adapter_kind=resolved_training_backend.adapter_kind,
    )

    return LocalTrainingRuntimeCompatibility(
        adapter_kind=resolved_training_backend.adapter_kind,
        training_backend_name=resolved_training_backend.backend_name,
        example_generation_backend_name=training_example_backend.backend_name,
        evidence_backend_name=evidence_backend.backend_name,
        scorer_backend_name=scorer_backend.backend_name,
        acceptance_policy_name=resolved_acceptance_policy.policy_name,
        privacy_guard_name=resolved_privacy_guard.guard_name,
    )


def _require_adapter_kind_support(
    *,
    component_type: str,
    component_name: str,
    supported_adapter_kinds: tuple[str, ...],
    adapter_kind: str,
) -> None:
    normalized_supported = tuple(
        value.strip().lower() for value in supported_adapter_kinds
    )
    normalized_adapter_kind = adapter_kind.strip().lower()
    if (
        ANY_ADAPTER_KIND in normalized_supported
        or normalized_adapter_kind in normalized_supported
    ):
        return
    raise ValueError(
        f"Incompatible {component_type}: {component_name} does not support "
        f"adapter_kind={adapter_kind}."
    )
