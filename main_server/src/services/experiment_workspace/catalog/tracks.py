"""Experiment workspace track spec와 builder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from main_server.src.services.experiment_workspace.catalog import (
    federated_runtime_sections as fl_catalog_sections,
)
from main_server.src.services.experiment_workspace.catalog.artifact_items import (
    build_generated_initial_checkpoint_items,
    build_generated_query_source_items,
)
from main_server.src.services.experiment_workspace.catalog.build_context import (
    ExperimentCatalogBuildContext,
)
from main_server.src.services.experiment_workspace.catalog.constants import (
    AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
    CENTRAL_ADAPTATION_RUNTIME_PATH,
    FEDERATED_SIMULATION_RUNTIME_PATH,
    MAIN_SERVER_ROUND_RUNTIME_PATH,
    SEED_RUNTIME_PATH,
)
from main_server.src.services.experiment_workspace.catalog.metadata import (
    CatalogCoreMethodResolver,
    CatalogMetadataResolver,
    CatalogTagResolver,
    build_dataset_preset_metadata,
    build_federated_run_preset_metadata,
    resolve_catalog_item_name,
    string_or_none,
)
from main_server.src.services.experiment_workspace.catalog.section_builders import (
    build_config_group_section,
    build_entrypoint_section,
)
from main_server.src.services.experiment_workspace.payloads import (
    CatalogItemPayload,
    CatalogSectionPayload,
    CatalogTrackPayload,
)

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
    selector_group: str | None = None
    core_method_resolver: CatalogCoreMethodResolver | None = None
    metadata_keys: tuple[str, ...] | None = None
    tag_resolver: CatalogTagResolver | None = None
    metadata_resolver: CatalogMetadataResolver | None = None
    extra_items_builder: (
        Callable[[ExperimentCatalogBuildContext], tuple[CatalogItemPayload, ...]] | None
    ) = None


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
    display_name="기준선 생성",
    description="분류기와 프로토타입 기준 자산을 만드는 시작 단계입니다.",
    supported_runtime_paths=(SEED_RUNTIME_PATH,),
    sections=(
        EntrypointSectionSpec(
            section_name="entrypoints",
            display_name="실행 작업",
            description="이 탭에서 바로 실행할 수 있는 작업 목록입니다.",
            relative_paths=(
                "conf/entrypoints/central_classifier_seed/train_softmax_classifier.yaml",
                "conf/entrypoints/prototype_pack/seed_prototypes.yaml",
            ),
            supported_runtime_paths=(SEED_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="dataset_presets",
            display_name="데이터셋",
            description="기준선 생성 단계에서 바로 쓰는 dataset alias입니다.",
            relative_dir="conf/execution_context/dataset_asset",
            item_kind="hydra_preset",
            family_name="dataset",
            preset_group="dataset",
            selector_group="execution_context/dataset_asset",
            supported_runtime_paths=(SEED_RUNTIME_PATH,),
            metadata_resolver=build_dataset_preset_metadata,
        ),
        ConfigGroupSectionSpec(
            section_name="embedding_presets",
            display_name="임베딩",
            description="분류기/프로토타입 생성에 쓰는 임베딩 preset입니다.",
            relative_dir="conf/execution_context/embedding_adapter",
            item_kind="hydra_preset",
            family_name="embedding",
            preset_group="embedding",
            selector_group="execution_context/embedding_adapter",
            supported_runtime_paths=(SEED_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="runtime_presets",
            display_name="실행 환경",
            description="기준선 생성 단계에서 공통으로 쓰는 runtime preset입니다.",
            relative_dir="conf/execution_context/runtime_env",
            item_kind="hydra_preset",
            family_name="runtime",
            preset_group="runtime",
            selector_group="execution_context/runtime_env",
            supported_runtime_paths=(SEED_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="prototype_builders",
            display_name="프로토타입 빌더",
            description="프로토타입 pack을 어떤 빌더로 생성할지 정합니다.",
            relative_dir="conf/strategy_axes/prototype/build_strategy",
            item_kind="hydra_preset",
            family_name="prototype_pack",
            preset_group="prototype_builder",
            selector_group="strategy_axes/prototype/build_strategy",
            supported_runtime_paths=(SEED_RUNTIME_PATH,),
            core_method_resolver=lambda _path, raw: resolve_catalog_item_name(raw),
        ),
    ),
)

CENTRAL_ADAPTATION_TRACK_SPEC = CatalogTrackSpec(
    track_name="central_adaptation",
    display_name="중앙 적응 비교",
    description=(
        "서버 한 곳에서 적응 방법론을 비교하는 단계입니다. 현재는 LoRA 기반이 "
        "기본이지만, 같은 축에서 DoRA 같은 PEFT 변형도 확장할 수 있게 둡니다."
    ),
    supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
    sections=(
        EntrypointSectionSpec(
            section_name="entrypoints",
            display_name="실행 작업",
            description="중앙 적응 비교선에서 직접 실행할 작업 목록입니다.",
            relative_paths=(
                "conf/entrypoints/central_ssl_control/train_lora_classifier.yaml",
                "conf/entrypoints/central_ssl_control/train_lora_pseudo_label_classifier.yaml",
                "conf/entrypoints/central_ssl_control/train_lora_fixmatch.yaml",
                "conf/entrypoints/central_ssl_control/train_lora_bootstrap_classifier_teacher.yaml",
            ),
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="dataset_presets",
            display_name="데이터셋",
            description="중앙 적응 비교선에서 쓰는 dataset alias입니다.",
            relative_dir="conf/execution_context/dataset_asset",
            item_kind="hydra_preset",
            family_name="dataset",
            preset_group="dataset",
            selector_group="execution_context/dataset_asset",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
            metadata_resolver=build_dataset_preset_metadata,
        ),
        ConfigGroupSectionSpec(
            section_name="runtime_presets",
            display_name="실행 환경",
            description="중앙 적응 실행에 쓰는 runtime preset입니다.",
            relative_dir="conf/execution_context/runtime_env",
            item_kind="hydra_preset",
            family_name="runtime",
            preset_group="runtime",
            selector_group="execution_context/runtime_env",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="paper_backbones",
            display_name="백본",
            description="중앙 적응에 쓰는 backbone preset입니다.",
            relative_dir="conf/strategy_axes/adaptation/transformer_backbone",
            item_kind="hydra_preset",
            family_name="backbone",
            preset_group="paper_backbone",
            selector_group="strategy_axes/adaptation/transformer_backbone",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
            metadata_keys=("model_id", "revision", "pooling", "max_length"),
        ),
        ConfigGroupSectionSpec(
            section_name="peft_methods",
            display_name="PEFT 방법",
            description=(
                "적응에 쓰는 PEFT preset입니다. 현재는 LoRA가 기본이지만 같은 "
                "축에서 DoRA 같은 변형을 추가할 수 있습니다."
            ),
            relative_dir="conf/strategy_axes/adaptation/peft_adapter",
            item_kind="hydra_preset",
            family_name="peft_adapter",
            preset_group="lora",
            selector_group="strategy_axes/adaptation/peft_adapter",
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
            display_name="적응 실행 프리셋",
            description="중앙 적응 baseline에서 공통으로 쓰는 실행 preset입니다.",
            relative_dir="conf/track_presets/central_ssl_control/training_preset",
            item_kind="hydra_preset",
            family_name="run_preset",
            preset_group="lora_run_preset",
            selector_group="track_presets/central_ssl_control/training_preset",
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
            section_name="query_sources",
            display_name="Query 데이터 소스",
            description=(
                "supervised, teacher bootstrap, FixMatch가 공유하는 train/"
                "unlabeled source preset입니다."
            ),
            relative_dir="conf/track_presets/central_ssl_control/query_source",
            item_kind="hydra_preset",
            family_name="train_source",
            preset_group="query_source",
            selector_group="track_presets/central_ssl_control/query_source",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
            extra_items_builder=lambda context: build_generated_query_source_items(
                repo_root=context.repo_root,
                relative_repo_path=context.relative_repo_path,
            ),
        ),
        ConfigGroupSectionSpec(
            section_name="pseudo_label_algorithms",
            display_name="의사라벨 선택 방식",
            description=(
                "중앙 pseudo-label adaptation에서 쓰는 selection "
                "algorithm preset입니다."
            ),
            relative_dir="conf/strategy_axes/ssl/pseudo_label_selection",
            item_kind="hydra_preset",
            family_name="pseudo_label_algorithm",
            preset_group="pseudo_label_algorithm",
            selector_group="strategy_axes/ssl/pseudo_label_selection",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
            metadata_keys=(
                "confidence_threshold",
                "margin_threshold",
                "algorithm_name",
            ),
        ),
        ConfigGroupSectionSpec(
            section_name="query_ssl_methods",
            display_name="Query SSL 목표 함수",
            description="중앙 query SSL objective preset입니다.",
            relative_dir="conf/strategy_axes/ssl/consistency_method",
            item_kind="hydra_preset",
            family_name="ssl_method",
            preset_group="query_ssl_method",
            selector_group="strategy_axes/ssl/consistency_method",
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
                ("requires_multiview",) if bool(raw.get("require_multiview")) else ()
            ),
        ),
        ConfigGroupSectionSpec(
            section_name="query_ssl_augmenters",
            display_name="멀티뷰 증강",
            description="중앙 FixMatch/query SSL multiview augmenter preset입니다.",
            relative_dir="conf/strategy_axes/ssl/augmentation",
            item_kind="hydra_preset",
            family_name="ssl_augmenter",
            preset_group="query_ssl_augmenter",
            selector_group="strategy_axes/ssl/augmentation",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="initial_checkpoints",
            display_name="초기 체크포인트",
            description="중앙 적응 시작 시 warm-start checkpoint preset입니다.",
            relative_dir="conf/strategy_axes/adaptation/initial_checkpoint",
            item_kind="hydra_preset",
            family_name="initial_checkpoint",
            preset_group="query_adaptation_initial_checkpoint",
            selector_group="strategy_axes/adaptation/initial_checkpoint",
            supported_runtime_paths=(CENTRAL_ADAPTATION_RUNTIME_PATH,),
            extra_items_builder=lambda context: (
                build_generated_initial_checkpoint_items(
                    repo_root=context.repo_root,
                    relative_repo_path=context.relative_repo_path,
                )
            ),
        ),
    ),
)

FEDERATED_RUNTIME_TRACK_SPEC = CatalogTrackSpec(
    track_name="federated_runtime",
    display_name="연합 런타임 점검",
    description="여러 agent가 참여하는 FL 조합과 round runtime을 점검하는 단계입니다.",
    supported_runtime_paths=(
        FEDERATED_SIMULATION_RUNTIME_PATH,
        MAIN_SERVER_ROUND_RUNTIME_PATH,
        AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
    ),
    sections=(
        EntrypointSectionSpec(
            section_name="entrypoints",
            display_name="실행 작업",
            description="현재 FL baseline을 직접 실행하는 작업 목록입니다.",
            relative_paths=("conf/entrypoints/fl_ssl/run_federated_simulation.yaml",),
            supported_runtime_paths=(
                FEDERATED_SIMULATION_RUNTIME_PATH,
                MAIN_SERVER_ROUND_RUNTIME_PATH,
            ),
        ),
        ConfigGroupSectionSpec(
            section_name="dataset_presets",
            display_name="데이터셋",
            description="FL simulation에 쓰는 dataset alias입니다.",
            relative_dir="conf/execution_context/dataset_asset",
            item_kind="hydra_preset",
            family_name="dataset",
            preset_group="dataset",
            selector_group="execution_context/dataset_asset",
            supported_runtime_paths=(FEDERATED_SIMULATION_RUNTIME_PATH,),
            metadata_resolver=build_dataset_preset_metadata,
        ),
        ConfigGroupSectionSpec(
            section_name="embedding_presets",
            display_name="임베딩",
            description="FL simulation에서 쓰는 embedding preset입니다.",
            relative_dir="conf/execution_context/embedding_adapter",
            item_kind="hydra_preset",
            family_name="embedding",
            preset_group="embedding",
            selector_group="execution_context/embedding_adapter",
            supported_runtime_paths=(FEDERATED_SIMULATION_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="runtime_presets",
            display_name="실행 환경",
            description="FL simulation runtime preset입니다.",
            relative_dir="conf/execution_context/runtime_env",
            item_kind="hydra_preset",
            family_name="runtime",
            preset_group="runtime",
            selector_group="execution_context/runtime_env",
            supported_runtime_paths=(FEDERATED_SIMULATION_RUNTIME_PATH,),
        ),
        ConfigGroupSectionSpec(
            section_name="federated_run_presets",
            display_name="연합 실행 프리셋",
            description=(
                "client 수, rounds, max_examples 같은 FL simulation preset입니다."
            ),
            relative_dir="conf/track_presets/fl_ssl/simulation_preset",
            item_kind="hydra_preset",
            family_name="federated_run_preset",
            preset_group="federated_run_preset",
            selector_group="track_presets/fl_ssl/simulation_preset",
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
            display_name="프로토타입 빌더",
            description="FL baseline prototype rebuild/build preset입니다.",
            relative_dir="conf/strategy_axes/prototype/build_strategy",
            item_kind="hydra_preset",
            family_name="prototype_pack",
            preset_group="prototype_builder",
            selector_group="strategy_axes/prototype/build_strategy",
            supported_runtime_paths=(FEDERATED_SIMULATION_RUNTIME_PATH,),
            core_method_resolver=lambda _path, raw: resolve_catalog_item_name(raw),
        ),
        CustomSectionSpec(
            build_section=lambda context: (
                fl_catalog_sections.build_local_update_profile_section(context)
            )
        ),
        ConfigGroupSectionSpec(
            section_name="round_runtime_profiles",
            display_name="라운드 런타임 프로필",
            description="FL server round의 adapter family와 aggregation 조합입니다.",
            relative_dir="conf/strategy_axes/fl/round_runtime_profile",
            item_kind="round_runtime_profile",
            family_name="round_runtime",
            preset_group="round_runtime_profile",
            selector_group="strategy_axes/fl/round_runtime_profile",
            supported_runtime_paths=(
                FEDERATED_SIMULATION_RUNTIME_PATH,
                MAIN_SERVER_ROUND_RUNTIME_PATH,
            ),
            metadata_keys=(
                "adapter_family_name",
                "aggregation_backend_name",
                "classifier_head_bootstrap_logit_scale",
            ),
        ),
        CustomSectionSpec(
            build_section=lambda context: (
                fl_catalog_sections.build_adapter_family_catalog_section(context)
            )
        ),
        CustomSectionSpec(
            build_section=lambda context: (
                fl_catalog_sections.build_aggregation_backend_section(context)
            )
        ),
        CustomSectionSpec(
            build_section=lambda context: (
                fl_catalog_sections.build_training_backend_section(context)
            )
        ),
        CustomSectionSpec(
            build_section=lambda context: (
                fl_catalog_sections.build_training_example_backend_section(context)
            )
        ),
        CustomSectionSpec(
            build_section=lambda context: (
                fl_catalog_sections.build_evidence_backend_section(context)
            )
        ),
        CustomSectionSpec(
            build_section=lambda context: (
                fl_catalog_sections.build_scoring_backend_section(context)
            )
        ),
        CustomSectionSpec(
            build_section=lambda context: (
                fl_catalog_sections.build_acceptance_policy_section(context)
            )
        ),
        CustomSectionSpec(
            build_section=lambda context: (
                fl_catalog_sections.build_privacy_guard_section(context)
            )
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
        _build_track_payload(spec, context=context) for spec in EXPERIMENT_TRACK_SPECS
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
            selector_group=spec.selector_group,
            supported_runtime_paths=spec.supported_runtime_paths,
            iter_yaml_files=context.iter_yaml_files,
            load_yaml_mapping=context.load_yaml_mapping,
            relative_repo_path=context.relative_repo_path,
            core_method_resolver=spec.core_method_resolver,
            metadata_keys=spec.metadata_keys,
            tag_resolver=spec.tag_resolver,
            metadata_resolver=spec.metadata_resolver,
            extra_items=(
                ()
                if spec.extra_items_builder is None
                else spec.extra_items_builder(context)
            ),
        )
    return spec.build_section(context)
