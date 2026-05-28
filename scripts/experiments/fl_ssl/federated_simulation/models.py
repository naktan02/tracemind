"""Federated simulation용 설정/결과 모델."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.federated_ssl.capability_plan import FederatedSslCapabilityPlan
from methods.federated_ssl.execution_plan import FederatedSslExecutionPlan
from methods.federated_ssl.local_update_profile import LocalUpdateProfile
from scripts.experiments.fl_ssl.federated_simulation import (
    simulation_result_models,
)
from scripts.runtime_adapters.federated_server.task_config_surface import (
    FederatedTrainingTaskConfig,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec

FL_DATA_SOURCE_RUNTIME_SPLIT_FROM_TRAIN = "runtime_split_from_train"
FL_DATA_SOURCE_MATERIALIZED_CLIENT_SPLIT = "materialized_client_split"
FL_DATA_SOURCE_MODES = frozenset(
    {
        FL_DATA_SOURCE_RUNTIME_SPLIT_FROM_TRAIN,
        FL_DATA_SOURCE_MATERIALIZED_CLIENT_SPLIT,
    }
)

ClientEvaluationSummary = simulation_result_models.ClientEvaluationSummary
ClientRoundSummary = simulation_result_models.ClientRoundSummary
SimulationEvaluation = simulation_result_models.SimulationEvaluation
SimulationResult = simulation_result_models.SimulationResult
SimulationRoundSummary = simulation_result_models.SimulationRoundSummary


@dataclass(slots=True)
class FederatedClientShard:
    """한 client에 할당된 train row 묶음."""

    client_id: str
    rows: list[LabeledQueryRow]
    labeled_rows: list[LabeledQueryRow] = field(default_factory=list)
    unlabeled_rows: list[LabeledQueryRow] = field(default_factory=list)
    client_pool_split_enforced: bool = False


@dataclass(slots=True)
class FederatedDatasetSplit:
    """bootstrap과 client shard로 나눈 train subset."""

    bootstrap_rows: list[LabeledQueryRow]
    client_shards: tuple[FederatedClientShard, ...]


@dataclass(slots=True)
class FederatedValidationConfig:
    """validation score/acceptance 계산 설정."""

    similarity_name: str
    scorer_backend_name: str
    score_policy_name: str | None
    confidence_threshold: float
    margin_threshold: float
    score_top_k: int | None = None


@dataclass(slots=True)
class FederatedDiagnosticsConfig:
    """selection dump 저장 설정."""

    dump_dir_name: str


@dataclass(frozen=True, slots=True)
class FederatedArtifactPersistenceConfig:
    """simulation artifact 사본 저장 정책."""

    persist_agent_local_updates: bool = False

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object] | None,
    ) -> "FederatedArtifactPersistenceConfig":
        if source is None:
            return cls()
        return cls(
            persist_agent_local_updates=bool(
                source.get("persist_agent_local_updates", False)
            )
        )


@dataclass(frozen=True, slots=True)
class FederatedResumeConfig:
    """round 단위 simulation 재개 정책."""

    checkpoint_enabled: bool = True
    enabled: bool = False
    run_dir: str | None = None

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object] | None,
    ) -> "FederatedResumeConfig":
        if source is None:
            return cls()
        return cls(
            checkpoint_enabled=bool(source.get("checkpoint_enabled", True)),
            enabled=bool(source.get("enabled", False)),
            run_dir=_optional_str(source.get("run_dir")),
        )

    def __post_init__(self) -> None:
        if self.enabled and self.run_dir is None:
            raise ValueError("resume.run_dir is required when resume.enabled=true.")


@dataclass(frozen=True, slots=True)
class FederatedDiagnosticViewConfig:
    """client-local pseudo-label diagnostics에 쓸 unlabeled subset 설정."""

    enabled: bool = True
    selection_policy: str = "deterministic_random"
    max_rows: int = 512
    seed_offset: int = 1309

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object] | None,
    ) -> "FederatedDiagnosticViewConfig":
        if source is None:
            return cls()
        return cls(
            enabled=bool(source.get("enabled", True)),
            selection_policy=str(
                source.get("selection_policy", "deterministic_random")
            ).strip(),
            max_rows=int(source.get("max_rows", 512)),
            seed_offset=int(source.get("seed_offset", 1309)),
        )

    def __post_init__(self) -> None:
        if self.selection_policy != "deterministic_random":
            raise ValueError(
                "diagnostic_view.selection_policy currently supports "
                "deterministic_random only."
            )
        if self.max_rows <= 0:
            raise ValueError("diagnostic_view.max_rows must be positive.")


@dataclass(frozen=True, slots=True)
class FederatedFinalProjectionConfig:
    """최종 global LoRA representation projection artifact 설정."""

    enabled: bool = True
    dataset_names: tuple[str, ...] = ("validation", "test")
    fail_on_error: bool = False

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object] | None,
    ) -> "FederatedFinalProjectionConfig":
        if source is None:
            return cls()
        raw_dataset_names = source.get("dataset_names", ("validation", "test"))
        if isinstance(raw_dataset_names, str):
            dataset_names = tuple(
                name.strip() for name in raw_dataset_names.split(",") if name.strip()
            )
        else:
            dataset_names = tuple(
                str(name).strip() for name in raw_dataset_names if str(name).strip()
            )
        return cls(
            enabled=bool(source.get("enabled", True)),
            dataset_names=dataset_names or ("validation", "test"),
            fail_on_error=bool(source.get("fail_on_error", False)),
        )

    def __post_init__(self) -> None:
        allowed_names = {"validation", "test"}
        unsupported_names = sorted(set(self.dataset_names) - allowed_names)
        if unsupported_names:
            raise ValueError(
                "final_projection.dataset_names only supports validation/test: "
                f"{unsupported_names}"
            )


@dataclass(slots=True)
class FederatedReportConfig:
    """paper 비교용 report schema 설정."""

    schema_version: str
    track: str
    table_role: str
    labeled_ratio: float
    unlabeled_ratio: float
    seed_count: int
    primary_metrics: list[str]
    secondary_metrics: list[str]


@dataclass(slots=True)
class FederatedClientPoolSplitConfig:
    """client별 local labeled/unlabeled pool 분할 설정."""

    labeled_ratio: float
    unlabeled_ratio: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.labeled_ratio <= 1.0:
            raise ValueError("client_pool_split.labeled_ratio must be between 0 and 1.")
        if not 0.0 <= self.unlabeled_ratio <= 1.0:
            raise ValueError(
                "client_pool_split.unlabeled_ratio must be between 0 and 1."
            )
        if abs((self.labeled_ratio + self.unlabeled_ratio) - 1.0) > 1e-9:
            raise ValueError(
                "client_pool_split labeled_ratio and unlabeled_ratio must sum to 1."
            )


@dataclass(slots=True)
class FederatedDataSourceConfig:
    """FL simulation이 client split을 어디서 가져왔는지 기록한다."""

    source_mode: str = FL_DATA_SOURCE_RUNTIME_SPLIT_FROM_TRAIN
    split_manifest_path: str | None = None
    split_manifest_sha256: str | None = None
    split_id: str | None = None
    source_selection: dict[str, object] = field(default_factory=dict)
    source_jsonl: dict[str, str] = field(default_factory=dict)
    labeled_policy: dict[str, object] = field(default_factory=dict)
    labeled_exposure_policy: dict[str, object] = field(default_factory=dict)
    view_schema: dict[str, object] = field(default_factory=dict)
    test_jsonl: str | None = None

    def __post_init__(self) -> None:
        if self.source_mode not in FL_DATA_SOURCE_MODES:
            raise ValueError(
                f"fl_data.source_mode must be one of {sorted(FL_DATA_SOURCE_MODES)}."
            )
        if (
            self.source_mode == FL_DATA_SOURCE_MATERIALIZED_CLIENT_SPLIT
            and not self.split_manifest_path
        ):
            raise ValueError(
                "fl_data.split_manifest is required when source_mode is "
                "materialized_client_split."
            )


@dataclass(slots=True)
class FederatedSslMethodConfig:
    """FL SSL descriptor 선택과 report metadata 설정."""

    schema_version: str
    name: str
    display_name: str
    method_role: str
    implementation_status: str
    original_source: dict[str, object] = field(default_factory=dict)
    scenario: str | None = None
    use_original_parameters: bool = True
    local_budget_policy: str = "iteration_capped"
    original_parameters: dict[str, object] = field(default_factory=dict)
    parameter_overrides: dict[str, object] = field(default_factory=dict)
    effective_parameters: dict[str, object] = field(default_factory=dict)
    parameter_override_status: str = "original"
    trace_mapping: dict[str, object] = field(default_factory=dict)
    client_step: dict[str, object] = field(default_factory=dict)
    server_step: dict[str, object] = field(default_factory=dict)
    round_state_exchange: dict[str, object] = field(default_factory=dict)
    report_tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


@dataclass(slots=True)
class FederatedRoundRuntimeConfig:
    """simulation이 사용할 update family와 v1 payload compatibility 설정."""

    adapter_family_name: str
    aggregation_backend_name: str
    update_family_name: str
    runtime_payload_key: str | None = None
    runtime_payloads: dict[str, object] = field(default_factory=dict)
    round_runtime_payload_builder: str | None = None
    local_objective_executors: tuple[str, ...] = ()
    initial_state_builder: str | None = None
    validation_evaluator: str | None = None
    final_projection_builder: str | None = None
    transient_resource_cleaner: str | None = None

    def __post_init__(self) -> None:
        normalized_adapter_family = (
            self.adapter_family_name.strip()
            .lower()
            .replace(
                "-",
                "_",
            )
        )
        if not normalized_adapter_family:
            raise ValueError("round_runtime.adapter_family_name must not be empty.")
        normalized_update_family = (
            self.update_family_name.strip()
            .lower()
            .replace(
                "-",
                "_",
            )
        )
        if not normalized_update_family:
            raise ValueError("round_runtime.update_family_name must not be empty.")
        self.adapter_family_name = normalized_adapter_family
        self.update_family_name = normalized_update_family
        normalized_runtime_payload_key = _optional_str(self.runtime_payload_key)
        if normalized_runtime_payload_key is not None:
            normalized_runtime_payload_key = (
                normalized_runtime_payload_key.lower().replace("-", "_")
            )
        self.runtime_payload_key = normalized_runtime_payload_key

    @property
    def payload_adapter_kind(self) -> str:
        """v1 shared payload/aggregation compatibility용 adapter kind."""

        return self.adapter_family_name

    def runtime_payload_for_update_family(self) -> object | None:
        """update-family config가 지정한 runtime payload를 반환한다."""

        if self.runtime_payload_key is not None:
            return self.runtime_payloads.get(self.runtime_payload_key)
        return self.runtime_payloads.get(self.update_family_name)


@dataclass(frozen=True, slots=True)
class FederatedQuerySslObjectiveConfig:
    """manual FL 조합이 실제 Query SSL algorithm을 가리키는지 나타내는 설정."""

    method_name: str
    algorithm_name: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    strong_view_policy: str = "first_aug"
    unlabeled_batch_size: int | None = None

    @classmethod
    def from_objective_config(
        cls,
        objective_config: TrainingObjectiveConfig | None,
    ) -> "FederatedQuerySslObjectiveConfig | None":
        """Training objective extras에서 query_ssl.* 축을 읽는다."""

        if objective_config is None:
            return None
        extras = objective_config.get_component_extras("query_ssl")
        method_name = _optional_str(extras.get("method_name"))
        algorithm_name = _optional_str(extras.get("algorithm_name"))
        if method_name is None and algorithm_name is None:
            return None
        if method_name is None or algorithm_name is None:
            raise ValueError(
                "query_ssl objective extras require both method_name and "
                "algorithm_name."
            )
        unlabeled_batch_size_raw = extras.get("unlabeled_batch_size")
        unlabeled_batch_size = (
            None if unlabeled_batch_size_raw is None else int(unlabeled_batch_size_raw)
        )
        if unlabeled_batch_size is not None and unlabeled_batch_size <= 0:
            raise ValueError("query_ssl.unlabeled_batch_size must be positive.")
        return cls(
            method_name=method_name,
            algorithm_name=algorithm_name,
            parameters={},
            strong_view_policy=(
                _optional_str(extras.get("strong_view_policy")) or "first_aug"
            ),
            unlabeled_batch_size=unlabeled_batch_size,
        )

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object],
        *,
        strong_view_policy: str = "first_aug",
    ) -> "FederatedQuerySslObjectiveConfig":
        """Hydra query_ssl_method mapping을 typed config로 해석한다."""

        method_name = _optional_str(source.get("name"))
        algorithm_name = _optional_str(source.get("algorithm_name"))
        if method_name is None or algorithm_name is None:
            raise ValueError("query_ssl_method requires name and algorithm_name.")
        parameters = {
            str(key): value
            for key, value in source.items()
            if str(key) not in {"name", "algorithm_name"}
        }
        unlabeled_batch_size = parameters.get("unlabeled_batch_size")
        return cls(
            method_name=method_name,
            algorithm_name=algorithm_name,
            parameters=parameters,
            strong_view_policy=strong_view_policy,
            unlabeled_batch_size=(
                None if unlabeled_batch_size is None else int(unlabeled_batch_size)
            ),
        )


@dataclass(frozen=True, slots=True)
class FederatedLocalTrainerRuntimeConfig:
    """FL simulation client trainer가 transformer/PEFT stack을 여는 방식."""

    device: str = "cpu"
    local_files_only: bool = True
    cache_dir: str = "hf_cache"
    trust_remote_code: bool = False
    classifier_dropout: float = 0.1


@dataclass(frozen=True, slots=True)
class FederatedPeerProbeConfig:
    """peer helper selection vector를 만들 fixed probe surface 설정."""

    enabled: bool = True
    selection_policy: str = "label_balanced"
    max_rows: int = 128
    seed_offset: int = 907

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object] | None,
    ) -> "FederatedPeerProbeConfig":
        if source is None:
            return cls()
        return cls(
            enabled=bool(source.get("enabled", True)),
            selection_policy=str(
                source.get("selection_policy", "label_balanced")
            ).strip(),
            max_rows=int(source.get("max_rows", 128)),
            seed_offset=int(source.get("seed_offset", 907)),
        )

    def __post_init__(self) -> None:
        if self.selection_policy != "label_balanced":
            raise ValueError(
                "peer_probe.selection_policy currently supports label_balanced only."
            )
        if self.max_rows <= 0:
            raise ValueError("peer_probe.max_rows must be positive.")


@dataclass(frozen=True, slots=True)
class FederatedPeerProbeManifest:
    """고정 probe subset 재현에 필요한 report metadata."""

    selection_policy: str
    seed: int
    seed_offset: int
    source: str
    requested_max_rows: int
    row_count: int
    query_ids_sha256: str
    label_distribution: dict[str, int]
    query_ids: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "metadata_status": "recorded",
            "selection_policy": self.selection_policy,
            "seed": self.seed,
            "seed_offset": self.seed_offset,
            "source": self.source,
            "requested_max_rows": self.requested_max_rows,
            "row_count": self.row_count,
            "query_ids_sha256": self.query_ids_sha256,
            "label_distribution": dict(self.label_distribution),
            "query_ids": list(self.query_ids),
        }


@dataclass(slots=True)
class SimulationRunRequest:
    """FL SSL simulation 한 번을 실행하는 typed request."""

    train_rows: list[LabeledQueryRow]
    validation_rows: list[LabeledQueryRow]
    output_dir: Path
    client_count: int
    rounds: int
    bootstrap_ratio: float
    seed: int
    embedding_spec: EmbeddingAdapterSpec
    model_id: str
    training_scope: str
    round_runtime_config: FederatedRoundRuntimeConfig
    shard_policy: FederatedShardPolicyConfig
    training_task_config: FederatedTrainingTaskConfig
    validation_config: FederatedValidationConfig
    diagnostics_config: FederatedDiagnosticsConfig
    test_rows: list[LabeledQueryRow] = field(default_factory=list)
    artifact_persistence_config: FederatedArtifactPersistenceConfig = field(
        default_factory=FederatedArtifactPersistenceConfig
    )
    resume_config: FederatedResumeConfig = field(default_factory=FederatedResumeConfig)
    run_budget_name: str | None = None
    run_output_dir: str | None = None
    ssl_method_config: FederatedSslMethodConfig | None = None
    client_pool_split_config: FederatedClientPoolSplitConfig | None = None
    materialized_dataset_split: FederatedDatasetSplit | None = None
    data_source_config: FederatedDataSourceConfig = field(
        default_factory=FederatedDataSourceConfig
    )
    report_config: FederatedReportConfig | None = None
    local_update_profile: LocalUpdateProfile | None = None
    execution_plan: FederatedSslExecutionPlan | None = None
    capability_plan: FederatedSslCapabilityPlan | None = None
    server_step_executor: str | None = None
    query_ssl_objective_config: FederatedQuerySslObjectiveConfig | None = None
    local_trainer_runtime_config: FederatedLocalTrainerRuntimeConfig = field(
        default_factory=FederatedLocalTrainerRuntimeConfig
    )
    diagnostic_view_config: FederatedDiagnosticViewConfig = field(
        default_factory=FederatedDiagnosticViewConfig
    )
    final_projection_config: FederatedFinalProjectionConfig = field(
        default_factory=FederatedFinalProjectionConfig
    )
    peer_probe_config: FederatedPeerProbeConfig = field(
        default_factory=FederatedPeerProbeConfig
    )
