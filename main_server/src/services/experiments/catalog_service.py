"""개발자용 read-only experiment catalog service."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

from agent.src.services.inference.scoring_backends import (
    list_scoring_backend_catalog_entries,
)
from agent.src.services.training.acceptance_policies.registry import (
    list_pseudo_label_acceptance_policy_catalog_entries,
)
from agent.src.services.training.evidence_backends.registry import (
    list_pseudo_label_evidence_backend_catalog_entries,
)
from agent.src.services.training.input_backends.registry import (
    list_training_example_backend_catalog_entries,
)
from agent.src.services.training.privacy_guard_service import (
    list_shared_adapter_privacy_guard_catalog_entries,
)
from agent.src.services.training.runtime_compatibility import (
    validate_live_agent_stored_event_runtime,
    validate_local_training_runtime,
)
from agent.src.services.training.training_backends.registry import (
    list_shared_adapter_training_backend_catalog_entries,
)
from main_server.src.services.experiments.catalog_constants import (
    AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
    CENTRAL_ADAPTATION_RUNTIME_PATH,
    FEDERATED_SIMULATION_RUNTIME_PATH,
    MAIN_SERVER_ROUND_RUNTIME_PATH,
    SEED_RUNTIME_PATH,
)
from main_server.src.services.experiments.catalog_metadata import (
    build_dataset_preset_metadata,
    build_federated_run_preset_metadata,
    declared_fields,
    extract_override_fields,
    extract_scalar_metadata,
    resolve_catalog_item_name,
    string_or_none,
)
from main_server.src.services.experiments.catalog_section_builders import (
    build_adapter_family_section,
    build_config_group_section,
    build_entrypoint_section,
    build_registry_section,
)
from main_server.src.services.experiments.payloads import (
    CatalogItemPayload,
    CatalogSectionPayload,
    CatalogTrackPayload,
    ExperimentCatalogPayload,
)
from main_server.src.services.rounds.aggregation_service import (
    list_shared_adapter_aggregation_backend_catalog_entries,
)
from shared.src.config.adapter_family_metadata import (
    list_shared_adapter_family_metadata,
)
from shared.src.config.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.config.training_defaults import (
    DEFAULT_TRAINING_PROFILE,
    build_default_secure_aggregation_config,
    build_default_training_selection_policy,
)
from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingTaskPayload,
)


class ExperimentCatalogService:
    """현재 코드/설정에서 읽어오는 read-only 전략 catalog."""

    def __init__(self, *, repo_root: Path | None = None) -> None:
        self._repo_root = repo_root or Path(__file__).resolve().parents[4]

    def build_catalog(self) -> ExperimentCatalogPayload:
        """현재 저장소 상태를 반영한 catalog payload를 조립한다."""

        return ExperimentCatalogPayload(
            generated_at=datetime.now(tz=timezone.utc),
            source_root=str(self._repo_root),
            tracks=(
                self._build_seed_track(),
                self._build_central_adaptation_track(),
                self._build_federated_runtime_track(),
            ),
        )

    def _build_entrypoint_section(
        self,
        *,
        section_name: str,
        display_name: str,
        description: str,
        relative_paths: tuple[str, ...],
        supported_runtime_paths: tuple[str, ...],
    ) -> CatalogSectionPayload:
        return build_entrypoint_section(
            section_name=section_name,
            display_name=display_name,
            description=description,
            relative_paths=relative_paths,
            supported_runtime_paths=supported_runtime_paths,
            repo_root=self._repo_root,
            load_yaml_mapping=self._load_yaml_mapping,
            relative_repo_path=self._relative_repo_path,
            resolve_script_path=self._resolve_script_path,
        )

    def _build_config_group_section(
        self,
        *,
        section_name: str,
        display_name: str,
        description: str,
        relative_dir: str,
        item_kind: str,
        family_name: str,
        preset_group: str,
        supported_runtime_paths: tuple[str, ...],
        core_method_resolver=None,
        metadata_keys: tuple[str, ...] | None = None,
        tag_resolver=None,
        metadata_resolver=None,
    ) -> CatalogSectionPayload:
        return build_config_group_section(
            section_name=section_name,
            display_name=display_name,
            description=description,
            relative_dir=relative_dir,
            item_kind=item_kind,
            family_name=family_name,
            preset_group=preset_group,
            supported_runtime_paths=supported_runtime_paths,
            iter_yaml_files=self._iter_yaml_files,
            load_yaml_mapping=self._load_yaml_mapping,
            relative_repo_path=self._relative_repo_path,
            core_method_resolver=core_method_resolver,
            metadata_keys=metadata_keys,
            tag_resolver=tag_resolver,
            metadata_resolver=metadata_resolver,
        )

    def _build_seed_track(self) -> CatalogTrackPayload:
        supported_runtime_paths = (SEED_RUNTIME_PATH,)
        return CatalogTrackPayload(
            track_name="seed",
            display_name="Seed",
            description=("classifier/prototype seed 자산을 만드는 baseline track."),
            entrypoint_section_name="entrypoints",
            supported_runtime_paths=supported_runtime_paths,
            sections=(
                self._build_entrypoint_section(
                    section_name="entrypoints",
                    display_name="Entrypoints",
                    description="현재 seed 단계에서 직접 실행 가능한 Hydra job 목록.",
                    relative_paths=(
                        "scripts/conf/experiments/train_softmax_classifier.yaml",
                        "scripts/conf/prototypes/seed_prototypes.yaml",
                    ),
                    supported_runtime_paths=supported_runtime_paths,
                ),
                self._build_config_group_section(
                    section_name="dataset_presets",
                    display_name="Dataset Presets",
                    description="seed 단계에서 바로 재사용하는 dataset alias.",
                    relative_dir="scripts/conf/dataset",
                    item_kind="hydra_preset",
                    family_name="dataset",
                    preset_group="dataset",
                    supported_runtime_paths=supported_runtime_paths,
                    metadata_resolver=build_dataset_preset_metadata,
                ),
                self._build_config_group_section(
                    section_name="embedding_presets",
                    display_name="Embedding Presets",
                    description=(
                        "seed classifier/prototype build에 쓰는 "
                        "embedding preset."
                    ),
                    relative_dir="scripts/conf/embedding",
                    item_kind="hydra_preset",
                    family_name="embedding",
                    preset_group="embedding",
                    supported_runtime_paths=supported_runtime_paths,
                ),
                self._build_config_group_section(
                    section_name="runtime_presets",
                    display_name="Runtime Presets",
                    description="seed 단계에서 공통으로 쓰는 runtime preset.",
                    relative_dir="scripts/conf/runtime",
                    item_kind="hydra_preset",
                    family_name="runtime",
                    preset_group="runtime",
                    supported_runtime_paths=supported_runtime_paths,
                ),
                self._build_config_group_section(
                    section_name="prototype_builders",
                    display_name="Prototype Builders",
                    description=(
                        "prototype pack을 어떤 builder로 생성할지 "
                        "정하는 preset."
                    ),
                    relative_dir="scripts/conf/prototype_builder",
                    item_kind="hydra_preset",
                    family_name="prototype_pack",
                    preset_group="prototype_builder",
                    core_method_resolver=lambda _path, raw: resolve_catalog_item_name(
                        raw
                    ),
                    supported_runtime_paths=supported_runtime_paths,
                ),
            ),
        )

    def _build_central_adaptation_track(self) -> CatalogTrackPayload:
        supported_runtime_paths = (CENTRAL_ADAPTATION_RUNTIME_PATH,)
        return CatalogTrackPayload(
            track_name="central_adaptation",
            display_name="Central Adaptation",
            description="중앙 LoRA/supervised/SSL 비교선을 조합하는 track.",
            entrypoint_section_name="entrypoints",
            supported_runtime_paths=supported_runtime_paths,
            sections=(
                self._build_entrypoint_section(
                    section_name="entrypoints",
                    display_name="Entrypoints",
                    description=(
                        "현재 중앙 적응 비교선에서 직접 실행 가능한 "
                        "Hydra job 목록."
                    ),
                    relative_paths=(
                        "scripts/conf/experiments/train_lora_classifier.yaml",
                        "scripts/conf/experiments/train_lora_pseudo_label_classifier.yaml",
                        "scripts/conf/experiments/train_lora_fixmatch.yaml",
                        "scripts/conf/experiments/train_lora_bootstrap_classifier_teacher.yaml",
                    ),
                    supported_runtime_paths=supported_runtime_paths,
                ),
                self._build_config_group_section(
                    section_name="dataset_presets",
                    display_name="Dataset Presets",
                    description="중앙 적응 비교선에서 쓰는 dataset alias.",
                    relative_dir="scripts/conf/dataset",
                    item_kind="hydra_preset",
                    family_name="dataset",
                    preset_group="dataset",
                    supported_runtime_paths=supported_runtime_paths,
                    metadata_resolver=build_dataset_preset_metadata,
                ),
                self._build_config_group_section(
                    section_name="runtime_presets",
                    display_name="Runtime Presets",
                    description="중앙 적응 실행 runtime preset.",
                    relative_dir="scripts/conf/runtime",
                    item_kind="hydra_preset",
                    family_name="runtime",
                    preset_group="runtime",
                    supported_runtime_paths=supported_runtime_paths,
                ),
                self._build_config_group_section(
                    section_name="paper_backbones",
                    display_name="Paper Backbones",
                    description="중앙 적응용 backbone preset.",
                    relative_dir="scripts/conf/paper_backbone",
                    item_kind="hydra_preset",
                    family_name="backbone",
                    preset_group="paper_backbone",
                    supported_runtime_paths=supported_runtime_paths,
                    metadata_keys=("model_id", "revision", "pooling", "max_length"),
                ),
                self._build_config_group_section(
                    section_name="peft_methods",
                    display_name="PEFT Methods",
                    description="현재 중앙 적응에서 쓰는 LoRA 계열 preset.",
                    relative_dir="scripts/conf/lora",
                    item_kind="hydra_preset",
                    family_name="peft_adapter",
                    core_method_resolver=lambda _path, _raw: "lora",
                    preset_group="lora",
                    supported_runtime_paths=supported_runtime_paths,
                    metadata_keys=(
                        "rank",
                        "alpha",
                        "dropout",
                        "bias",
                        "target_modules",
                    ),
                ),
                self._build_config_group_section(
                    section_name="lora_run_presets",
                    display_name="LoRA Run Presets",
                    description="중앙 LoRA baseline 공통 runtime preset.",
                    relative_dir="scripts/conf/lora_run_preset",
                    item_kind="hydra_preset",
                    family_name="run_preset",
                    preset_group="lora_run_preset",
                    supported_runtime_paths=supported_runtime_paths,
                    metadata_keys=(
                        "seed",
                        "train_batch_size",
                        "eval_batch_size",
                        "epochs",
                        "learning_rate",
                        "classifier_learning_rate",
                    ),
                ),
                self._build_config_group_section(
                    section_name="lora_train_sources",
                    display_name="LoRA Train Sources",
                    description=(
                        "supervised/teacher bootstrap에서 쓰는 "
                        "중앙 train source preset."
                    ),
                    relative_dir="scripts/conf/lora_train_source",
                    item_kind="hydra_preset",
                    family_name="train_source",
                    preset_group="lora_train_source",
                    supported_runtime_paths=supported_runtime_paths,
                ),
                self._build_config_group_section(
                    section_name="query_ssl_train_sources",
                    display_name="Query SSL Train Sources",
                    description=(
                        "pseudo-label/FixMatch query adaptation용 "
                        "train source preset."
                    ),
                    relative_dir="scripts/conf/query_ssl_train_source",
                    item_kind="hydra_preset",
                    family_name="train_source",
                    preset_group="query_ssl_train_source",
                    supported_runtime_paths=supported_runtime_paths,
                ),
                self._build_config_group_section(
                    section_name="bootstrap_teacher_sources",
                    display_name="Bootstrap Teacher Sources",
                    description="teacher bootstrap seed source preset.",
                    relative_dir="scripts/conf/bootstrap_teacher_source",
                    item_kind="hydra_preset",
                    family_name="bootstrap_teacher_source",
                    preset_group="bootstrap_teacher_source",
                    supported_runtime_paths=supported_runtime_paths,
                ),
                self._build_config_group_section(
                    section_name="pseudo_label_algorithms",
                    display_name="Pseudo-label Algorithms",
                    description=(
                        "중앙 pseudo-label adaptation에서 쓰는 "
                        "selection algorithm preset."
                    ),
                    relative_dir="scripts/conf/pseudo_label_algorithm",
                    item_kind="hydra_preset",
                    family_name="pseudo_label_algorithm",
                    preset_group="pseudo_label_algorithm",
                    supported_runtime_paths=supported_runtime_paths,
                    metadata_keys=(
                        "confidence_threshold",
                        "margin_threshold",
                        "algorithm_name",
                    ),
                ),
                self._build_config_group_section(
                    section_name="query_ssl_methods",
                    display_name="Query SSL Methods",
                    description="중앙 query SSL objective preset.",
                    relative_dir="scripts/conf/query_ssl_method",
                    item_kind="hydra_preset",
                    family_name="ssl_method",
                    core_method_resolver=lambda _path, raw: string_or_none(
                        raw.get("algorithm_name")
                    ),
                    preset_group="query_ssl_method",
                    supported_runtime_paths=supported_runtime_paths,
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
                self._build_config_group_section(
                    section_name="query_ssl_augmenters",
                    display_name="Query SSL Augmenters",
                    description="중앙 FixMatch/query SSL multiview augmenter preset.",
                    relative_dir="scripts/conf/query_ssl_augmenter",
                    item_kind="hydra_preset",
                    family_name="ssl_augmenter",
                    preset_group="query_ssl_augmenter",
                    supported_runtime_paths=supported_runtime_paths,
                ),
                self._build_config_group_section(
                    section_name="initial_checkpoints",
                    display_name="Initial Checkpoints",
                    description="중앙 적응 시작 시 warm-start checkpoint preset.",
                    relative_dir="scripts/conf/query_adaptation_initial_checkpoint",
                    item_kind="hydra_preset",
                    family_name="initial_checkpoint",
                    preset_group="query_adaptation_initial_checkpoint",
                    supported_runtime_paths=supported_runtime_paths,
                ),
            ),
        )

    def _build_federated_runtime_track(self) -> CatalogTrackPayload:
        supported_runtime_paths = (
            FEDERATED_SIMULATION_RUNTIME_PATH,
            MAIN_SERVER_ROUND_RUNTIME_PATH,
            AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
        )
        return CatalogTrackPayload(
            track_name="federated_runtime",
            display_name="Federated Runtime",
            description="현재 시스템 FL baseline과 round runtime 조합 inventory.",
            entrypoint_section_name="entrypoints",
            supported_runtime_paths=supported_runtime_paths,
            sections=(
                self._build_entrypoint_section(
                    section_name="entrypoints",
                    display_name="Entrypoints",
                    description="현재 FL baseline을 직접 실행하는 Hydra job 목록.",
                    relative_paths=(
                        "scripts/conf/experiments/run_federated_simulation.yaml",
                    ),
                    supported_runtime_paths=(
                        FEDERATED_SIMULATION_RUNTIME_PATH,
                        MAIN_SERVER_ROUND_RUNTIME_PATH,
                    ),
                ),
                self._build_config_group_section(
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
                self._build_config_group_section(
                    section_name="embedding_presets",
                    display_name="Embedding Presets",
                    description="FL simulation에서 쓰는 embedding preset.",
                    relative_dir="scripts/conf/embedding",
                    item_kind="hydra_preset",
                    family_name="embedding",
                    preset_group="embedding",
                    supported_runtime_paths=(FEDERATED_SIMULATION_RUNTIME_PATH,),
                ),
                self._build_config_group_section(
                    section_name="runtime_presets",
                    display_name="Runtime Presets",
                    description="FL simulation runtime preset.",
                    relative_dir="scripts/conf/runtime",
                    item_kind="hydra_preset",
                    family_name="runtime",
                    preset_group="runtime",
                    supported_runtime_paths=(FEDERATED_SIMULATION_RUNTIME_PATH,),
                ),
                self._build_config_group_section(
                    section_name="federated_run_presets",
                    display_name="Federated Run Presets",
                    description=(
                        "client 수, rounds, max_examples 같은 "
                        "FL simulation preset."
                    ),
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
                self._build_config_group_section(
                    section_name="prototype_builders",
                    display_name="Prototype Builders",
                    description="FL baseline prototype rebuild/build preset.",
                    relative_dir="scripts/conf/prototype_builder",
                    item_kind="hydra_preset",
                    family_name="prototype_pack",
                    preset_group="prototype_builder",
                    core_method_resolver=lambda _path, raw: resolve_catalog_item_name(
                        raw
                    ),
                    supported_runtime_paths=(FEDERATED_SIMULATION_RUNTIME_PATH,),
                ),
                self._build_training_algorithm_profile_section(),
                self._build_adapter_family_section(),
                self._build_aggregation_backend_section(),
                self._build_training_backend_section(),
                self._build_training_example_backend_section(),
                self._build_evidence_backend_section(),
                self._build_scoring_backend_section(),
                self._build_acceptance_policy_section(),
                self._build_privacy_guard_section(),
            ),
        )

    def _build_training_algorithm_profile_section(self) -> CatalogSectionPayload:
        items: list[CatalogItemPayload] = []
        for path in self._iter_yaml_files("scripts/conf/training_algorithm_profile"):
            raw = self._load_yaml_mapping(path)
            profile_name = (
                string_or_none(raw.get("algorithm_profile_name")) or path.stem
            )
            objective_config = TrainingObjectiveConfig.from_mapping(raw)
            training_task = self._build_catalog_training_task(
                training_scope=string_or_none(raw.get("training_scope"))
                or "adapter_only",
                objective_config=objective_config,
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
                    source_of_truth=self._relative_repo_path(path),
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

    def _build_adapter_family_section(self) -> CatalogSectionPayload:
        return build_adapter_family_section(
            family_metadata=list_shared_adapter_family_metadata(),
            source_of_truth_for_module=self._source_of_truth_for_module,
            supported_runtime_paths=(
                FEDERATED_SIMULATION_RUNTIME_PATH,
                MAIN_SERVER_ROUND_RUNTIME_PATH,
                AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
            ),
        )

    def _build_aggregation_backend_section(self) -> CatalogSectionPayload:
        return build_registry_section(
            section_name="aggregation_backends",
            display_name="Aggregation Backends",
            item_kind="aggregation_backend",
            description="adapter family별 서버 aggregation backend.",
            source_module_name="main_server.src.services.rounds.aggregation_service",
            entries=list_shared_adapter_aggregation_backend_catalog_entries(),
            source_of_truth_for_module=self._source_of_truth_for_module,
            supported_runtime_paths=(
                FEDERATED_SIMULATION_RUNTIME_PATH,
                MAIN_SERVER_ROUND_RUNTIME_PATH,
            ),
        )

    def _build_training_backend_section(self) -> CatalogSectionPayload:
        return build_registry_section(
            section_name="training_backends",
            display_name="Training Backends",
            item_kind="training_backend",
            description="로컬 accepted example을 update payload로 바꾸는 backend.",
            source_module_name=(
                "agent.src.services.training.training_backends.registry"
            ),
            entries=list_shared_adapter_training_backend_catalog_entries(),
            source_of_truth_for_module=self._source_of_truth_for_module,
            supported_runtime_paths=(
                FEDERATED_SIMULATION_RUNTIME_PATH,
                AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
            ),
        )

    def _build_training_example_backend_section(self) -> CatalogSectionPayload:
        return build_registry_section(
            section_name="example_generation_backends",
            display_name="Example Generation Backends",
            item_kind="example_generation_backend",
            description=(
                "source row 또는 stored event를 학습 예시로 "
                "재구성하는 backend."
            ),
            source_module_name="agent.src.services.training.input_backends.registry",
            entries=list_training_example_backend_catalog_entries(),
            source_of_truth_for_module=self._source_of_truth_for_module,
            runtime_path_resolver=self._resolve_example_generation_runtime_paths,
        )

    def _build_evidence_backend_section(self) -> CatalogSectionPayload:
        return build_registry_section(
            section_name="evidence_backends",
            display_name="Evidence Backends",
            item_kind="evidence_backend",
            description="ScoredEvent를 pseudo-label evidence로 정규화하는 backend.",
            source_module_name=(
                "agent.src.services.training.evidence_backends.registry"
            ),
            entries=list_pseudo_label_evidence_backend_catalog_entries(),
            source_of_truth_for_module=self._source_of_truth_for_module,
            supported_runtime_paths=(
                FEDERATED_SIMULATION_RUNTIME_PATH,
                AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
            ),
        )

    def _build_scoring_backend_section(self) -> CatalogSectionPayload:
        return build_registry_section(
            section_name="scoring_backends",
            display_name="Scoring Backends",
            item_kind="scoring_backend",
            description=(
                "embedding/prototype/shared_state로 category score를 "
                "계산하는 backend."
            ),
            source_module_name="agent.src.services.inference.scoring_backends",
            entries=list_scoring_backend_catalog_entries(),
            source_of_truth_for_module=self._source_of_truth_for_module,
            runtime_path_resolver=self._resolve_scoring_backend_runtime_paths,
        )

    def _build_acceptance_policy_section(self) -> CatalogSectionPayload:
        return build_registry_section(
            section_name="acceptance_policies",
            display_name="Acceptance Policies",
            item_kind="acceptance_policy",
            description="pseudo-label evidence를 accepted candidate로 해석하는 정책.",
            source_module_name=(
                "agent.src.services.training.acceptance_policies.registry"
            ),
            entries=list_pseudo_label_acceptance_policy_catalog_entries(),
            source_of_truth_for_module=self._source_of_truth_for_module,
            supported_runtime_paths=(
                FEDERATED_SIMULATION_RUNTIME_PATH,
                AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
            ),
        )

    def _build_privacy_guard_section(self) -> CatalogSectionPayload:
        return build_registry_section(
            section_name="privacy_guards",
            display_name="Privacy Guards",
            item_kind="privacy_guard",
            description="local update 보호 계층 registry.",
            source_module_name="agent.src.services.training.privacy_guard_service",
            entries=list_shared_adapter_privacy_guard_catalog_entries(),
            source_of_truth_for_module=self._source_of_truth_for_module,
            supported_runtime_paths=(
                FEDERATED_SIMULATION_RUNTIME_PATH,
                AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
            ),
        )

    def _resolve_example_generation_runtime_paths(
        self,
        entry: RegistryCatalogEntry,
    ) -> tuple[str, ...]:
        runtime_paths = [FEDERATED_SIMULATION_RUNTIME_PATH]
        if bool(entry.metadata.get("supports_stored_event_rebuild")):
            runtime_paths.append(AGENT_LIVE_STORED_EVENT_RUNTIME_PATH)
        return tuple(runtime_paths)

    def _resolve_scoring_backend_runtime_paths(
        self,
        entry: RegistryCatalogEntry,
    ) -> tuple[str, ...]:
        runtime_paths = [FEDERATED_SIMULATION_RUNTIME_PATH]
        if not bool(entry.metadata.get("requires_shared_state")):
            runtime_paths.append(AGENT_LIVE_STORED_EVENT_RUNTIME_PATH)
        return tuple(runtime_paths)

    def _build_catalog_training_task(
        self,
        *,
        training_scope: str,
        objective_config: TrainingObjectiveConfig,
    ) -> TrainingTaskPayload:
        return TrainingTaskPayload(
            task_id="catalog_preview_task",
            round_id="catalog_preview_round",
            model_id="catalog_preview_model",
            model_revision="catalog_preview_revision",
            task_type=TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING,
            training_scope=training_scope,
            local_epochs=DEFAULT_TRAINING_PROFILE.local_epochs,
            batch_size=DEFAULT_TRAINING_PROFILE.batch_size,
            learning_rate=DEFAULT_TRAINING_PROFILE.learning_rate,
            max_steps=DEFAULT_TRAINING_PROFILE.max_steps,
            objective_config=objective_config,
            selection_policy=build_default_training_selection_policy(),
            min_required_examples=DEFAULT_TRAINING_PROFILE.min_required_examples,
            gradient_clip_norm=DEFAULT_TRAINING_PROFILE.gradient_clip_norm,
            secure_aggregation=build_default_secure_aggregation_config(),
        )

    def _iter_yaml_files(self, relative_dir: str) -> tuple[Path, ...]:
        root = self._repo_root / relative_dir
        return tuple(sorted(root.glob("*.yaml")))

    def load_config_group_item(
        self,
        *,
        relative_dir: str,
        item_name: str,
    ) -> dict[str, object]:
        """Hydra config group item 하나를 직접 읽는다."""

        path = self._repo_root / relative_dir / f"{item_name}.yaml"
        if not path.exists():
            raise ValueError(
                f"Unknown config group item: dir={relative_dir}, item={item_name}."
            )
        return self._load_yaml_mapping(path)

    def load_relative_yaml_mapping(self, relative_path: str) -> dict[str, object]:
        """repo root 기준 상대 경로 YAML mapping을 읽는다."""

        path = self._repo_root / relative_path
        if not path.exists():
            raise ValueError(f"Unknown YAML path: {relative_path}.")
        return self._load_yaml_mapping(path)

    def _load_yaml_mapping(self, path: Path) -> dict[str, object]:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"Expected mapping config at {path}.")
        return dict(raw)

    def _resolve_script_path(self, job_config_path: str) -> str:
        if job_config_path.startswith("scripts/conf/experiments/"):
            return job_config_path.replace(
                "scripts/conf/experiments/",
                "scripts/experiments/",
            ).replace(
                ".yaml", ".py"
            )
        if job_config_path.startswith("scripts/conf/prototypes/"):
            return job_config_path.replace(
                "scripts/conf/prototypes/",
                "scripts/prototypes/",
            ).replace(
                ".yaml", ".py"
            )
        if job_config_path.startswith("scripts/conf/datasets/"):
            return job_config_path.replace(
                "scripts/conf/datasets/",
                "scripts/datasets/",
            ).replace(
                ".yaml", ".py"
            )
        raise ValueError(
            "Unsupported job config path for catalog script resolution: "
            f"{job_config_path}."
        )

    def _source_of_truth_for_module(self, module_name: str) -> str:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            return module_name
        return self._relative_repo_path(Path(spec.origin))

    def _relative_repo_path(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(self._repo_root))
        except ValueError:
            return str(resolved)
