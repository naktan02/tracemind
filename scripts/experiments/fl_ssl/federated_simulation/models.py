"""Federated simulation용 설정/결과 모델."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from methods.adaptation.lora_classifier.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.federated_ssl.execution_plan import FederatedSslExecutionPlan
from methods.federated_ssl.local_update_profile import LocalUpdateProfile
from methods.prototype.building.base import PrototypeBuildStrategy
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
class FederatedPrototypeRebuildConfig:
    """라운드별 prototype 재생성 메타데이터 설정."""

    embedding_backend: str
    mapping_version: str
    translation_model_id: str | None = None
    translation_model_revision: str | None = None
    translation_direction: str | None = None


@dataclass(slots=True)
class FederatedDiagnosticsConfig:
    """selection dump 저장 설정."""

    dump_dir_name: str


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
    client_step: dict[str, object] = field(default_factory=dict)
    server_step: dict[str, object] = field(default_factory=dict)
    round_state_exchange: dict[str, object] = field(default_factory=dict)
    report_tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FederatedLoraClassifierRuntimeConfig:
    """LoRA-classifier simulation bootstrap에 필요한 fixed scaffold snapshot."""

    training_backend_config: LoraClassifierTrainingBackendConfig
    artifact_format: str = "simulation_lora_classifier_state_ref"
    lora_adapter_artifact_ref: str | None = None
    classifier_head_artifact_ref: str | None = None

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object],
    ) -> "FederatedLoraClassifierRuntimeConfig":
        """Hydra round_runtime.lora_classifier mapping을 typed config로 해석한다."""

        artifact_format = str(
            source.get("artifact_format", "simulation_lora_classifier_state_ref")
        ).strip()
        if not artifact_format:
            raise ValueError("round_runtime.lora_classifier.artifact_format invalid.")
        return cls(
            training_backend_config=LoraClassifierTrainingBackendConfig.from_mapping(
                {
                    key: value
                    for key, value in source.items()
                    if key not in _LORA_CLASSIFIER_RUNTIME_ARTIFACT_KEYS
                }
            ),
            artifact_format=artifact_format,
            lora_adapter_artifact_ref=_optional_str(
                source.get("lora_adapter_artifact_ref")
            ),
            classifier_head_artifact_ref=_optional_str(
                source.get("classifier_head_artifact_ref")
            ),
        )

    def backbone_payload(self) -> dict[str, str | int]:
        """shared lora_classifier state에 넣을 backbone/tokenizer snapshot."""

        return self.training_backend_config.to_backbone_payload()

    def lora_config_payload(self) -> dict[str, str | int | float | bool]:
        """shared lora_classifier state에 넣을 LoRA config snapshot."""

        return self.training_backend_config.to_lora_config_payload()


_LORA_CLASSIFIER_RUNTIME_ARTIFACT_KEYS = frozenset(
    {
        "artifact_format",
        "lora_adapter_artifact_ref",
        "classifier_head_artifact_ref",
    }
)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


@dataclass(slots=True)
class FederatedRoundRuntimeConfig:
    """simulation이 사용할 shared family / aggregation backend 설정."""

    adapter_family_name: str
    aggregation_backend_name: str
    classifier_head_bootstrap_logit_scale: float = 8.0
    lora_classifier: FederatedLoraClassifierRuntimeConfig | None = None

    def runtime_payload_for_adapter_family(self) -> object | None:
        """adapter family 이름과 같은 runtime payload 필드를 돌려준다."""

        runtime_field_name = self.adapter_family_name.strip().lower().replace("-", "_")
        if not runtime_field_name:
            raise ValueError("round_runtime.adapter_family_name must not be empty.")
        return getattr(self, runtime_field_name, None)


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
    prototype_build_strategy: PrototypeBuildStrategy
    shard_policy: FederatedShardPolicyConfig
    training_task_config: FederatedTrainingTaskConfig
    validation_config: FederatedValidationConfig
    prototype_rebuild_config: FederatedPrototypeRebuildConfig
    diagnostics_config: FederatedDiagnosticsConfig
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
    query_ssl_objective_config: FederatedQuerySslObjectiveConfig | None = None
    local_trainer_runtime_config: FederatedLocalTrainerRuntimeConfig = field(
        default_factory=FederatedLocalTrainerRuntimeConfig
    )
