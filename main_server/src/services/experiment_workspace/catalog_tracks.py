"""Experiment workspace track spec와 builder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent.src.services.inference.scoring_backends import (
    list_scoring_backend_catalog_entries,
)
from agent.src.services.training.acceptance_policies.registry import (
    list_pseudo_label_acceptance_policy_catalog_entries,
)
from agent.src.services.training.backends.evidence.registry import (
    list_pseudo_label_evidence_backend_catalog_entries,
)
from agent.src.services.training.backends.inputs.registry import (
    list_training_example_backend_catalog_entries,
)
from agent.src.services.training.backends.training.registry import (
    list_shared_adapter_training_backend_catalog_entries,
)
from agent.src.services.training.execution.privacy_guard_service import (
    list_shared_adapter_privacy_guard_catalog_entries,
)
from agent.src.services.training.execution.runtime_compatibility import (
    validate_live_agent_stored_event_runtime,
    validate_local_training_runtime,
)
from main_server.src.services.experiment_workspace.catalog_build_context import (
    ExperimentCatalogBuildContext,
)
from main_server.src.services.experiment_workspace.catalog_constants import (
    AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
    CENTRAL_ADAPTATION_RUNTIME_PATH,
    FEDERATED_SIMULATION_RUNTIME_PATH,
    MAIN_SERVER_ROUND_RUNTIME_PATH,
    SEED_RUNTIME_PATH,
)
from main_server.src.services.experiment_workspace.catalog_metadata import (
    CatalogCoreMethodResolver,
    CatalogMetadataResolver,
    CatalogTagResolver,
    build_dataset_preset_metadata,
    build_federated_run_preset_metadata,
    declared_fields,
    extract_override_fields,
    extract_scalar_metadata,
    resolve_catalog_item_name,
    string_or_none,
)
from main_server.src.services.experiment_workspace.catalog_section_builders import (
    build_adapter_family_section,
    build_config_group_section,
    build_entrypoint_section,
    build_registry_section,
)
from main_server.src.services.experiment_workspace.payloads import (
    CatalogItemPayload,
    CatalogSectionPayload,
    CatalogTrackPayload,
)
from main_server.src.services.federation.rounds.aggregation.registry import (
    list_shared_adapter_aggregation_backend_catalog_entries,
)
from shared.src.config.adapter_family_metadata import (
    list_shared_adapter_family_metadata,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

BuildCustomSection = Callable[
    [ExperimentCatalogBuildContext],
    CatalogSectionPayload,
]


@dataclass(frozen=True, slots=True)
class EntrypointSectionSpec:
    section_name: str
    display_name: str
    description: str
    relative_paths: tuple[str, ...]
    supported_runtime_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConfigGroupSectionSpec:
    section_name: str
    display_name: str
    description: str
    relative_dir: str
    item_kind: str
    family_name: str
    preset_group: str
    supported_runtime_paths: tuple[str, ...]
    core_method_resolver: CatalogCoreMethodResolver | None = None
    metadata_keys: tuple[str, ...] | None = None
    tag_resolver: CatalogTagResolver | None = None
    metadata_resolver: CatalogMetadataResolver | None = None


@dataclass(frozen=True, slots=True)
class CustomSectionSpec:
    build_section: BuildCustomSection


CatalogSectionSpec = EntrypointSectionSpec | ConfigGroupSectionSpec | CustomSectionSpec


@dataclass(frozen=True, slots=True)
class CatalogTrackSpec:
    track_name: str
    display_name: str
    description: str
    supported_runtime_paths: tuple[str, ...]
    sections: tuple[CatalogSectionSpec, ...]
    entrypoint_section_name: str | None = "entrypoints"


SEED_TRACK_SPEC = CatalogTrackSpec(
    track_name="seed",
    display_name="Seed",
    description="classifier/prototype seed 자산을 만드는 baseline track.",
    supported_runtime_paths=(SEED_RUNTIME_PATH,),
    sections=(
        EntrypointSectionSpec(
            section_name="entrypoints",
            display_name="Entrypoints",
            description="현재 seed 단계에서 직접 실행 가능한 Hydra job 목록.",
            relative_paths=(
                "scripts/conf/experiments/train_softmax_classifier.yaml",
                "scripts/conf/prototypes/seed_prototypes.yaml",
            ),
            supported_runtime_paths=(SEED_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="dataset_presets",
            display_name="Dataset Presets",
            description="seed 단계에서 바로 재사용하는 dataset alias.",
            relative_dir="scripts/conf/dataset",
            item_kind="hydra_preset",
            family_name="dataset",
            preset_group="dataset",
            supported_runtime_paths=(SEED_RUNTIME_PATH,),
            metadata_resolver=build_dataset_preset_metadata,
        ),
        ConfigGroupSectionSpec(
            section_name="embedding_presets",
            display_name="Embedding Presets",
            description="seed classifier/prototype build에 쓰는 embedding preset.",
            relative_dir="scripts/conf/embedding",
            item_kind="hydra_preset",
            family_name="embedding",
            preset_group="embedding",
            supported_runtime_paths=(SEED_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="runtime_presets",
            display_name="Runtime Presets",
            description="seed 단계에서 공통으로 쓰는 runtime preset.",
            relative_dir="scripts/conf/runtime",
            item_kind="hydra_preset",
            family_name="runtime",
            preset_group="runtime",
            supported_runtime_paths=(SEED_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="prototype_builders",
            display_name="Prototype Builders",
            description="prototype pack을 어떤 builder로 생성할지 정하는 preset.",
            relative_dir="scripts/conf/prototype_builder",
            item_kind="hydra_preset",
            family_name="prototype_pack",
            preset_group="prototype_builder",
            supported_runtime_paths=(SEED_RUNTIME_PATH,),
            core_method_resolver=lambda _path, raw: resolve_catalog_item_name(raw),
        ),
    ),
)

CENTRAL_ADAPTATION_TRACK_SPEC = CatalogTrackSpec(
    track_name="central_adaptation",
    display_name="Central Adaptation",
    description="중앙 LoRA/supervised/SSL 비교선을 조합하는 track.",
    supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
    sections=(
        EntrypointSectionSpec(
            section_name="entrypoints",
            display_name="Entrypoints",
            description="현재 중앙 적응 비교선에서 직접 실행 가능한 Hydra job 목록.",
            relative_paths=(
                "scripts/conf/experiments/train_lora_classifier.yaml",
                "scripts/conf/experiments/train_lora_pseudo_label_classifier.yaml",
                "scripts/conf/experiments/train_lora_fixmatch.yaml",
                "scripts/conf/experiments/train_lora_bootstrap_classifier_teacher.yaml",
            ),
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="dataset_presets",
            display_name="Dataset Presets",
            description="중앙 적응 비교선에서 쓰는 dataset alias.",
            relative_dir="scripts/conf/dataset",
            item_kind="hydra_preset",
            family_name="dataset",
            preset_group="dataset",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
            metadata_resolver=build_dataset_preset_metadata,
        ),
        ConfigGroupSectionSpec(
            section_name="runtime_presets",
            display_name="Runtime Presets",
            description="중앙 적응 실행 runtime preset.",
            relative_dir="scripts/conf/runtime",
            item_kind="hydra_preset",
            family_name="runtime",
            preset_group="runtime",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="paper_backbones",
            display_name="Paper Backbones",
            description="중앙 적응용 backbone preset.",
            relative_dir="scripts/conf/paper_backbone",
            item_kind="hydra_preset",
            family_name="backbone",
            preset_group="paper_backbone",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
            metadata_keys=("model_id", "revision", "pooling", "max_length"),
        ),
        ConfigGroupSectionSpec(
            section_name="peft_methods",
            display_name="PEFT Methods",
            description="현재 중앙 적응에서 쓰는 LoRA 계열 preset.",
            relative_dir="scripts/conf/lora",
            item_kind="hydra_preset",
            family_name="peft_adapter",
            preset_group="lora",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
            core_method_resolver=lambda _path, _raw: "lora",
            metadata_keys=(
                "rank",
                "alpha",
                "dropout",
                "bias",
                "target_modules",
            ),
        ),
        ConfigGroupSectionSpec(
            section_name="lora_run_presets",
            display_name="LoRA Run Presets",
            description="중앙 LoRA baseline 공통 runtime preset.",
            relative_dir="scripts/conf/lora_run_preset",
            item_kind="hydra_preset",
            family_name="run_preset",
            preset_group="lora_run_preset",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
            metadata_keys=(
                "seed",
                "train_batch_size",
                "eval_batch_size",
                "epochs",
                "learning_rate",
                "classifier_learning_rate",
            ),
        ),
        ConfigGroupSectionSpec(
            section_name="lora_train_sources",
            display_name="LoRA Train Sources",
            description=(
                "supervised/teacher bootstrap에서 쓰는 중앙 train "
                "source preset."
            ),
            relative_dir="scripts/conf/lora_train_source",
            item_kind="hydra_preset",
            family_name="train_source",
            preset_group="lora_train_source",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="query_ssl_train_sources",
            display_name="Query SSL Train Sources",
            description="pseudo-label/FixMatch query adaptation용 train source preset.",
            relative_dir="scripts/conf/query_ssl_train_source",
            item_kind="hydra_preset",
            family_name="train_source",
            preset_group="query_ssl_train_source",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="bootstrap_teacher_sources",
            display_name="Bootstrap Teacher Sources",
            description="teacher bootstrap seed source preset.",
            relative_dir="scripts/conf/bootstrap_teacher_source",
            item_kind="hydra_preset",
            family_name="bootstrap_teacher_source",
            preset_group="bootstrap_teacher_source",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="pseudo_label_algorithms",
            display_name="Pseudo-label Algorithms",
            description=(
                "중앙 pseudo-label adaptation에서 쓰는 selection "
                "algorithm preset."
            ),
            relative_dir="scripts/conf/pseudo_label_algorithm",
            item_kind="hydra_preset",
            family_name="pseudo_label_algorithm",
            preset_group="pseudo_label_algorithm",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
            metadata_keys=(
                "confidence_threshold",
                "margin_threshold",
                "algorithm_name",
            ),
        ),
        ConfigGroupSectionSpec(
            section_name="query_ssl_methods",
            display_name="Query SSL Methods",
            description="중앙 query SSL objective preset.",
            relative_dir="scripts/conf/query_ssl_method",
            item_kind="hydra_preset",
            family_name="ssl_method",
            preset_group="query_ssl_method",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
            core_method_resolver=lambda _path, raw: string_or_none(
                raw.get("algorithm_name")
            ),
            metadata_keys=(
                "algorithm_name",
                "temperature",
                "p_cutoff",
                "hard_label",
                "lambda_u",
                "supervised_loss_weight",
                "require_multiview",
            ),
            tag_resolver=lambda _path, raw: (
                ("requires_multiview",)
                if bool(raw.get("require_multiview"))
                else ()
            ),
        ),
        ConfigGroupSectionSpec(
            section_name="query_ssl_augmenters",
            display_name="Query SSL Augmenters",
            description="중앙 FixMatch/query SSL multiview augmenter preset.",
            relative_dir="scripts/conf/query_ssl_augmenter",
            item_kind="hydra_preset",
            family_name="ssl_augmenter",
            preset_group="query_ssl_augmenter",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="initial_checkpoints",
            display_name="Initial Checkpoints",
            description="중앙 적응 시작 시 warm-start checkpoint preset.",
            relative_dir="scripts/conf/query_adaptation_initial_checkpoint",
            item_kind="hydra_preset",
            family_name="initial_checkpoint",
            preset_group="query_adaptation_initial_checkpoint",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
        ),
    ),
)

FEDERATED_RUNTIME_TRACK_SPEC = CatalogTrackSpec(
    track_name="federated_runtime",
    display_name="Federated Runtime",
    description="현재 시스템 FL baseline과 round runtime 조합 inventory.",
    supported_runtime_paths=(
        FEDERATED_SIMULATION_RUNTIME_PATH,
        MAIN_SERVER_ROUND_RUNTIME_PATH,
        AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
    ),
    sections=(
        EntrypointSectionSpec(
            section_name="entrypoints",
            display_name="Entrypoints",
            description="현재 FL baseline을 직접 실행하는 Hydra job 목록.",
            relative_paths=("scripts/conf/experiments/run_federated_simulation.yaml",),
            supported_runtime_paths=(
                FEDERATED_SIMULATION_RUNTIME_PATH,
                MAIN_SERVER_ROUND_RUNTIME_PATH,
            ),
        ),
        ConfigGroupSectionSpec(
            section_name="dataset_presets",
            display_name="Dataset Presets",
            description="FL simulation에 쓰는 dataset alias.",
            relative_dir="scripts/conf/dataset",
            item_kind="hydra_preset",
            family_name="dataset",
            preset_group="dataset",
            supported_runtime_paths=(FEDERATED_SIMULATION_RUNTIME_PATH,),
            metadata_resolver=build_dataset_preset_metadata,
        ),
        ConfigGroupSectionSpec(
            section_name="embedding_presets",
            display_name="Embedding Presets",
            description="FL simulation에서 쓰는 embedding preset.",
            relative_dir="scripts/conf/embedding",
            item_kind="hydra_preset",
            family_name="embedding",
            preset_group="embedding",
            supported_runtime_paths=(FEDERATED_SIMULATION_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="runtime_presets",
            display_name="Runtime Presets",
            description="FL simulation runtime preset.",
            relative_dir="scripts/conf/runtime",
            item_kind="hydra_preset",
            family_name="runtime",
            preset_group="runtime",
            supported_runtime_paths=(FEDERATED_SIMULATION_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="federated_run_presets",
            display_name="Federated Run Presets",
            description="client 수, rounds, max_examples 같은 FL simulation preset.",
            relative_dir="scripts/conf/federated_run_preset",
            item_kind="hydra_preset",
            family_name="federated_run_preset",
            preset_group="federated_run_preset",
            supported_runtime_paths=(FEDERATED_SIMULATION_RUNTIME_PATH,),
            metadata_keys=(
                "client_count",
                "rounds",
                "bootstrap_ratio",
                "max_examples",
                "min_required_examples",
            ),
            metadata_resolver=build_federated_run_preset_metadata,
        ),
        ConfigGroupSectionSpec(
            section_name="prototype_builders",
            display_name="Prototype Builders",
            description="FL baseline prototype rebuild/build preset.",
            relative_dir="scripts/conf/prototype_builder",
            item_kind="hydra_preset",
            family_name="prototype_pack",
            preset_group="prototype_builder",
            supported_runtime_paths=(FEDERATED_SIMULATION_RUNTIME_PATH,),
            core_method_resolver=lambda _path, raw: resolve_catalog_item_name(raw),
        ),
        CustomSectionSpec(
            build_section=lambda context: _build_training_algorithm_profile_section(
                context
            )
        ),
        CustomSectionSpec(
            build_section=lambda context: _build_adapter_family_section(context)
        ),
        CustomSectionSpec(
            build_section=lambda context: _build_aggregation_backend_section(context)
        ),
        CustomSectionSpec(
            build_section=lambda context: _build_training_backend_section(context)
        ),
        CustomSectionSpec(
            build_section=lambda context: _build_training_example_backend_section(
                context
            )
        ),
        CustomSectionSpec(
            build_section=lambda context: _build_evidence_backend_section(context)
        ),
        CustomSectionSpec(
            build_section=lambda context: _build_scoring_backend_section(context)
        ),
        CustomSectionSpec(
            build_section=lambda context: _build_acceptance_policy_section(context)
        ),
        CustomSectionSpec(
            build_section=lambda context: _build_privacy_guard_section(context)
        ),
    ),
)

EXPERIMENT_TRACK_SPECS = (
    SEED_TRACK_SPEC,
    CENTRAL_ADAPTATION_TRACK_SPEC,
    FEDERATED_RUNTIME_TRACK_SPEC,
)


def build_catalog_tracks(
    context: ExperimentCatalogBuildContext,
) -> tuple[CatalogTrackPayload, ...]:
    """정적 track spec을 현재 repo 상태 catalog payload로 직렬화한다."""

    return tuple(
        _build_track_payload(spec, context=context)
        for spec in EXPERIMENT_TRACK_SPECS
    )


def _build_track_payload(
    spec: CatalogTrackSpec,
    *,
    context: ExperimentCatalogBuildContext,
) -> CatalogTrackPayload:
    return CatalogTrackPayload(
        track_name=spec.track_name,
        display_name=spec.display_name,
        description=spec.description,
        entrypoint_section_name=spec.entrypoint_section_name,
        supported_runtime_paths=spec.supported_runtime_paths,
        sections=tuple(
            _build_section_payload(section_spec, context=context)
            for section_spec in spec.sections
        ),
    )


def _build_section_payload(
    spec: CatalogSectionSpec,
    *,
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    if isinstance(spec, EntrypointSectionSpec):
        return build_entrypoint_section(
            section_name=spec.section_name,
            display_name=spec.display_name,
            description=spec.description,
            relative_paths=spec.relative_paths,
            supported_runtime_paths=spec.supported_runtime_paths,
            repo_root=context.repo_root,
            load_yaml_mapping=context.load_yaml_mapping,
            relative_repo_path=context.relative_repo_path,
            resolve_script_path=context.resolve_script_path,
        )
    if isinstance(spec, ConfigGroupSectionSpec):
        return build_config_group_section(
            section_name=spec.section_name,
            display_name=spec.display_name,
            description=spec.description,
            relative_dir=spec.relative_dir,
            item_kind=spec.item_kind,
            family_name=spec.family_name,
            preset_group=spec.preset_group,
            supported_runtime_paths=spec.supported_runtime_paths,
            iter_yaml_files=context.iter_yaml_files,
            load_yaml_mapping=context.load_yaml_mapping,
            relative_repo_path=context.relative_repo_path,
            core_method_resolver=spec.core_method_resolver,
            metadata_keys=spec.metadata_keys,
            tag_resolver=spec.tag_resolver,
            metadata_resolver=spec.metadata_resolver,
        )
    return spec.build_section(context)


def _build_training_algorithm_profile_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    items: list[CatalogItemPayload] = []
    for path in context.iter_yaml_files("scripts/conf/training_algorithm_profile"):
        raw = context.load_yaml_mapping(path)
        profile_name = (
            string_or_none(raw.get("algorithm_profile_name")) or path.stem
        )
        objective_config = TrainingObjectiveConfig.from_mapping(raw)
        training_task = context.build_catalog_training_task(
            string_or_none(raw.get("training_scope")) or "adapter_only",
            objective_config,
        )
        runtime_paths = [FEDERATED_SIMULATION_RUNTIME_PATH]
        try:
            validate_local_training_runtime(training_task)
        except ValueError:
            pass
        else:
            runtime_paths.append(MAIN_SERVER_ROUND_RUNTIME_PATH)
        try:
            validate_live_agent_stored_event_runtime(training_task)
        except ValueError:
            pass
        else:
            runtime_paths.append(AGENT_LIVE_STORED_EVENT_RUNTIME_PATH)
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
        display_name="Training Algorithm Profiles",
        item_kind="training_algorithm_profile",
        description="현재 FL objective/aggregation 조합 preset.",
        source_of_truth="scripts/conf/training_algorithm_profile",
        source_kind="hydra_config_group",
        items=tuple(items),
    )


def _build_adapter_family_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    return build_adapter_family_section(
        family_metadata=list_shared_adapter_family_metadata(),
        source_of_truth_for_module=context.source_of_truth_for_module,
        supported_runtime_paths=(
            FEDERATED_SIMULATION_RUNTIME_PATH,
            MAIN_SERVER_ROUND_RUNTIME_PATH,
            AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
        ),
    )


def _build_aggregation_backend_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    return build_registry_section(
        section_name="aggregation_backends",
        display_name="Aggregation Backends",
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


def _build_training_backend_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    return build_registry_section(
        section_name="training_backends",
        display_name="Training Backends",
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


def _build_training_example_backend_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    return build_registry_section(
        section_name="example_generation_backends",
        display_name="Example Generation Backends",
        item_kind="example_generation_backend",
        description="source row 또는 stored event를 학습 예시로 재구성하는 backend.",
        source_module_name="agent.src.services.training.backends.inputs.registry",
        entries=list_training_example_backend_catalog_entries(),
        source_of_truth_for_module=context.source_of_truth_for_module,
        runtime_path_resolver=context.resolve_example_generation_runtime_paths,
    )


def _build_evidence_backend_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    return build_registry_section(
        section_name="evidence_backends",
        display_name="Evidence Backends",
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


def _build_scoring_backend_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    return build_registry_section(
        section_name="scoring_backends",
        display_name="Scoring Backends",
        item_kind="scoring_backend",
        description=(
            "embedding/prototype/shared_state로 category score를 계산하는 "
            "backend."
        ),
        source_module_name="agent.src.services.inference.scoring_backends",
        entries=list_scoring_backend_catalog_entries(),
        source_of_truth_for_module=context.source_of_truth_for_module,
        runtime_path_resolver=context.resolve_scoring_backend_runtime_paths,
    )


def _build_acceptance_policy_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    return build_registry_section(
        section_name="acceptance_policies",
        display_name="Acceptance Policies",
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


def _build_privacy_guard_section(
    context: ExperimentCatalogBuildContext,
) -> CatalogSectionPayload:
    return build_registry_section(
        section_name="privacy_guards",
        display_name="Privacy Guards",
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
