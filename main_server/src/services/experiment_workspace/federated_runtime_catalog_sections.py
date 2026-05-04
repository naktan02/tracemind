"""Federated runtime track의 registry/config catalog section builders."""

from __future__ import annotations

from main_server.src.services.experiment_workspace.catalog_build_context import (
    ExperimentCatalogBuildContext,
)
from main_server.src.services.experiment_workspace.catalog_constants import (
    AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
    FEDERATED_SIMULATION_RUNTIME_PATH,
    MAIN_SERVER_ROUND_RUNTIME_PATH,
)
from main_server.src.services.experiment_workspace.catalog_metadata import (
    declared_fields,
    extract_override_fields,
    extract_scalar_metadata,
    string_or_none,
)
from main_server.src.services.experiment_workspace.catalog_section_builders import (
    build_adapter_family_section,
    build_registry_section,
)
from main_server.src.services.experiment_workspace.payloads import (
    CatalogItemPayload,
    CatalogSectionPayload,
)
from main_server.src.services.federation.rounds.aggregation.registry import (
    list_shared_adapter_aggregation_backend_catalog_entries,
)
from shared.src.config.adapter_family_metadata import (
    list_shared_adapter_family_metadata,
)
from shared.src.config.local_training_registry_catalog import (
    ANY_ADAPTER_KIND,
    list_pseudo_label_acceptance_policy_catalog_entries,
    list_pseudo_label_evidence_backend_catalog_entries,
    list_scoring_backend_catalog_entries,
    list_shared_adapter_privacy_guard_catalog_entries,
    list_shared_adapter_training_backend_catalog_entries,
    list_training_example_backend_catalog_entries,
)
from shared.src.config.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.config.training_defaults import DEFAULT_TRAINING_PROFILE
from shared.src.contracts.training_contracts import TrainingObjectiveConfig


def build_training_algorithm_profile_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    """Hydra training profile preset을 runtime compatibility와 함께 노출한다."""

    items: list[CatalogItemPayload] = []
    for path in context.iter_yaml_files("scripts/conf/training_algorithm_profile"):
        raw = context.load_yaml_mapping(path)
        profile_name = string_or_none(raw.get("algorithm_profile_name")) or path.stem
        objective_config = TrainingObjectiveConfig.from_mapping(raw)
        training_task = context.build_catalog_training_task(
            string_or_none(raw.get("training_scope")) or "adapter_only",
            objective_config,
        )
        runtime_paths = _resolve_training_profile_runtime_paths(
            training_task.objective_config
        )
        items.append(
            CatalogItemPayload(
                item_name=profile_name,
                display_name=profile_name,
                item_kind="training_algorithm_profile",
                family_name=string_or_none(raw.get("adapter_family_name")),
                core_method_name=profile_name,
                variant_profile_name=profile_name,
                preset_group="training_algorithm_profile",
                source_of_truth=context.relative_repo_path(path),
                source_kind="hydra_config_group",
                compile_support="preset_selector",
                supported_adapter_kinds=(
                    string_or_none(raw.get("adapter_family_name")) or "",
                )
                if raw.get("adapter_family_name") is not None
                else (),
                supported_runtime_paths=tuple(runtime_paths),
                declared_fields=declared_fields(raw),
                override_fields=extract_override_fields(raw),
                metadata=extract_scalar_metadata(raw),
            )
        )
    return CatalogSectionPayload(
        section_name="training_algorithm_profiles",
        display_name="학습 알고리즘 프로필",
        item_kind="training_algorithm_profile",
        description="현재 FL objective/aggregation 조합 preset입니다.",
        source_of_truth="scripts/conf/training_algorithm_profile",
        source_kind="hydra_config_group",
        items=tuple(items),
    )


def build_adapter_family_catalog_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    """shared adapter family metadata를 FL runtime catalog에 노출한다."""

    return build_adapter_family_section(
        family_metadata=list_shared_adapter_family_metadata(),
        source_of_truth_for_module=context.source_of_truth_for_module,
        supported_runtime_paths=(
            FEDERATED_SIMULATION_RUNTIME_PATH,
            MAIN_SERVER_ROUND_RUNTIME_PATH,
            AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
        ),
    )


def build_aggregation_backend_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    """server aggregation backend registry를 FL runtime catalog에 노출한다."""

    return build_registry_section(
        section_name="aggregation_backends",
        display_name="집계 방식",
        item_kind="aggregation_backend",
        description="adapter family별 서버 aggregation backend.",
        source_module_name=(
            "main_server.src.services.federation.rounds.aggregation.registry"
        ),
        entries=list_shared_adapter_aggregation_backend_catalog_entries(),
        source_of_truth_for_module=context.source_of_truth_for_module,
        supported_runtime_paths=(
            FEDERATED_SIMULATION_RUNTIME_PATH,
            MAIN_SERVER_ROUND_RUNTIME_PATH,
        ),
    )


def build_training_backend_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    """agent local training backend registry를 FL runtime catalog에 노출한다."""

    return build_registry_section(
        section_name="training_backends",
        display_name="로컬 학습 백엔드",
        item_kind="training_backend",
        description="로컬 accepted example을 update payload로 바꾸는 backend.",
        source_module_name="agent.src.services.training.backends.training.registry",
        entries=list_shared_adapter_training_backend_catalog_entries(),
        source_of_truth_for_module=context.source_of_truth_for_module,
        supported_runtime_paths=(
            FEDERATED_SIMULATION_RUNTIME_PATH,
            AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
        ),
    )


def build_training_example_backend_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    """training example backend registry를 FL runtime catalog에 노출한다."""

    return build_registry_section(
        section_name="example_generation_backends",
        display_name="예제 생성 방식",
        item_kind="example_generation_backend",
        description="source row 또는 stored event를 학습 예시로 재구성하는 backend.",
        source_module_name="agent.src.services.training.backends.inputs.registry",
        entries=list_training_example_backend_catalog_entries(),
        source_of_truth_for_module=context.source_of_truth_for_module,
        runtime_path_resolver=context.resolve_example_generation_runtime_paths,
    )


def build_evidence_backend_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    """pseudo-label evidence backend registry를 FL runtime catalog에 노출한다."""

    return build_registry_section(
        section_name="evidence_backends",
        display_name="근거 정규화 방식",
        item_kind="evidence_backend",
        description="ScoredEvent를 pseudo-label evidence로 정규화하는 backend.",
        source_module_name="agent.src.services.training.backends.evidence.registry",
        entries=list_pseudo_label_evidence_backend_catalog_entries(),
        source_of_truth_for_module=context.source_of_truth_for_module,
        supported_runtime_paths=(
            FEDERATED_SIMULATION_RUNTIME_PATH,
            AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
        ),
    )


def build_scoring_backend_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    """scoring backend registry를 FL runtime catalog에 노출한다."""

    return build_registry_section(
        section_name="scoring_backends",
        display_name="점수 계산 방식",
        item_kind="scoring_backend",
        description=(
            "embedding/prototype/shared_state로 category score를 계산하는 backend."
        ),
        source_module_name="agent.src.services.inference.scoring_backends",
        entries=list_scoring_backend_catalog_entries(),
        source_of_truth_for_module=context.source_of_truth_for_module,
        runtime_path_resolver=context.resolve_scoring_backend_runtime_paths,
    )


def build_acceptance_policy_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    """pseudo-label acceptance policy registry를 FL runtime catalog에 노출한다."""

    return build_registry_section(
        section_name="acceptance_policies",
        display_name="채택 정책",
        item_kind="acceptance_policy",
        description="pseudo-label evidence를 accepted candidate로 해석하는 정책.",
        source_module_name="agent.src.services.training.acceptance_policies.registry",
        entries=list_pseudo_label_acceptance_policy_catalog_entries(),
        source_of_truth_for_module=context.source_of_truth_for_module,
        supported_runtime_paths=(
            FEDERATED_SIMULATION_RUNTIME_PATH,
            AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
        ),
    )


def build_privacy_guard_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    """local update privacy guard registry를 FL runtime catalog에 노출한다."""

    return build_registry_section(
        section_name="privacy_guards",
        display_name="보호 장치",
        item_kind="privacy_guard",
        description="local update 보호 계층 registry.",
        source_module_name="agent.src.services.training.execution.privacy_guard_service",
        entries=list_shared_adapter_privacy_guard_catalog_entries(),
        source_of_truth_for_module=context.source_of_truth_for_module,
        supported_runtime_paths=(
            FEDERATED_SIMULATION_RUNTIME_PATH,
            AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
        ),
    )


def _resolve_training_profile_runtime_paths(
    objective_config: TrainingObjectiveConfig,
) -> tuple[str, ...]:
    """Registry metadata만으로 objective 조합의 runtime compatibility를 계산한다."""

    runtime_paths = [FEDERATED_SIMULATION_RUNTIME_PATH]
    if not _is_local_training_runtime_catalog_compatible(objective_config):
        return tuple(runtime_paths)

    runtime_paths.append(MAIN_SERVER_ROUND_RUNTIME_PATH)
    example_backend = _find_catalog_entry(
        list_training_example_backend_catalog_entries(),
        objective_config.example_generation_backend_name
        or DEFAULT_TRAINING_PROFILE.example_generation_backend_name,
    )
    scorer_backend = _find_catalog_entry(
        list_scoring_backend_catalog_entries(),
        objective_config.scorer_backend_name
        or DEFAULT_TRAINING_PROFILE.scorer_backend_name,
    )
    if bool(example_backend.metadata.get("supports_stored_event_rebuild")) and (
        not bool(scorer_backend.metadata.get("requires_shared_state"))
    ):
        runtime_paths.append(AGENT_LIVE_STORED_EVENT_RUNTIME_PATH)
    return tuple(runtime_paths)


def _is_local_training_runtime_catalog_compatible(
    objective_config: TrainingObjectiveConfig,
) -> bool:
    """Registry metadata만으로 local training runtime 조합 호환성을 판정한다."""

    try:
        training_backend = _find_catalog_entry(
            list_shared_adapter_training_backend_catalog_entries(),
            objective_config.training_backend_name,
        )
        adapter_kind = _resolve_single_adapter_kind(training_backend)
        _require_catalog_adapter_kind_support(
            entry=_find_catalog_entry(
                list_training_example_backend_catalog_entries(),
                objective_config.example_generation_backend_name
                or DEFAULT_TRAINING_PROFILE.example_generation_backend_name,
            ),
            adapter_kind=adapter_kind,
        )
        _require_catalog_adapter_kind_support(
            entry=_find_catalog_entry(
                list_pseudo_label_evidence_backend_catalog_entries(),
                objective_config.evidence_backend_name
                or DEFAULT_TRAINING_PROFILE.evidence_backend_name,
            ),
            adapter_kind=adapter_kind,
        )
        _require_catalog_adapter_kind_support(
            entry=_find_catalog_entry(
                list_scoring_backend_catalog_entries(),
                objective_config.scorer_backend_name
                or DEFAULT_TRAINING_PROFILE.scorer_backend_name,
            ),
            adapter_kind=adapter_kind,
        )
        _require_catalog_adapter_kind_support(
            entry=_find_catalog_entry(
                list_pseudo_label_acceptance_policy_catalog_entries(),
                objective_config.acceptance_policy_name
                or DEFAULT_TRAINING_PROFILE.acceptance_policy_name,
            ),
            adapter_kind=adapter_kind,
        )
        _require_catalog_adapter_kind_support(
            entry=_find_catalog_entry(
                list_shared_adapter_privacy_guard_catalog_entries(),
                objective_config.privacy_guard_name
                or DEFAULT_TRAINING_PROFILE.privacy_guard_name,
            ),
            adapter_kind=adapter_kind,
        )
    except ValueError:
        return False
    return True


def _find_catalog_entry(
    entries: tuple[RegistryCatalogEntry, ...],
    item_name: str,
) -> RegistryCatalogEntry:
    normalized_name = item_name.strip().lower()
    for entry in entries:
        if entry.item_name.strip().lower() == normalized_name:
            return entry
    raise ValueError(f"Unknown registry catalog item: {item_name}.")


def _resolve_single_adapter_kind(entry: RegistryCatalogEntry) -> str:
    concrete_adapter_kinds = tuple(
        adapter_kind
        for adapter_kind in entry.supported_adapter_kinds
        if adapter_kind != ANY_ADAPTER_KIND
    )
    if len(concrete_adapter_kinds) != 1:
        raise ValueError(
            "Training backend catalog entry must expose one concrete "
            f"adapter kind: {entry.item_name}."
        )
    return concrete_adapter_kinds[0]


def _require_catalog_adapter_kind_support(
    *,
    entry: RegistryCatalogEntry,
    adapter_kind: str,
) -> None:
    normalized_supported = tuple(
        value.strip().lower() for value in entry.supported_adapter_kinds
    )
    normalized_adapter_kind = adapter_kind.strip().lower()
    if (
        ANY_ADAPTER_KIND in normalized_supported
        or normalized_adapter_kind in normalized_supported
    ):
        return
    raise ValueError(
        f"Catalog item does not support adapter_kind={adapter_kind}: {entry.item_name}."
    )
