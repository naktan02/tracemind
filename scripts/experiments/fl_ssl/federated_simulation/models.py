"""Federated simulation용 설정/결과 모델."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from methods.adaptation.lora_classifier.config import (
    LoraClassifierTrainingBackendConfig,
    build_lora_classifier_training_backend_config,
)
from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.federated_ssl.execution_plan import FederatedSslExecutionPlan
from methods.federated_ssl.local_update_profile import LocalUpdateProfile
from methods.prototype.building.base import PrototypeBuildStrategy
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
class ClientRoundSummary:
    """client 하나의 라운드 참여 요약."""

    client_id: str
    candidate_count: int
    accepted_count: int
    update_generated: bool
    delta_l2_norm: float | None = None
    aggregation_example_count: int | None = None
    client_train_time_seconds: float | None = None
    client_payload_bytes: int | None = None
    pseudo_label_confidence_mean: float | None = None
    pseudo_label_margin_mean: float | None = None
    pseudo_label_correct_count: int = 0
    pseudo_label_evaluated_count: int = 0
    accepted_label_distribution: dict[str, int] = field(default_factory=dict)
    rejected_label_distribution: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class SimulationEvaluation:
    """validation 평가 결과."""

    row_count: int
    top1_accuracy: float
    accepted_ratio: float
    loss: float = 0.0
    loss_kind: str = "not_computed"
    accuracy_top_1: float = 0.0
    correct_top_1: int = 0
    macro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    weighted_f1: float = 0.0
    balanced_accuracy: float = 0.0
    worst_category_f1: str | None = None
    worst_category_f1_value: float | None = None
    worst_category_recall: float | None = None
    worst_category_precision: float | None = None
    expected_calibration_error: float = 0.0
    max_calibration_error: float = 0.0
    overconfidence_gap: float = 0.0
    mean_true_label_probability: float = 0.0
    mean_top_1_probability: float = 0.0
    mean_margin_top1_top2: float = 0.0
    mean_correct_top_1_probability: float = 0.0
    mean_incorrect_top_1_probability: float = 0.0
    score_distribution_kind: str = "not_computed"
    selection_confidence_kind: str | None = None
    mean_selection_confidence: float = 0.0
    mean_selection_margin: float = 0.0
    per_label: dict[str, dict[str, int | float]] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    classification_report: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.accuracy_top_1 == 0.0 and self.top1_accuracy != 0.0:
            self.accuracy_top_1 = self.top1_accuracy


@dataclass(slots=True)
class ClientEvaluationSummary:
    """client별 heldout validation shard 평가 요약."""

    client_id: str
    validation: SimulationEvaluation


@dataclass(slots=True)
class SimulationRoundSummary:
    """한 라운드 종료 후 요약."""

    round_id: str
    model_revision: str
    prototype_version: str
    update_count: int
    validation: SimulationEvaluation
    clients: tuple[ClientRoundSummary, ...]
    round_time_seconds: float | None = None
    total_payload_bytes: int | None = None
    gpu_memory_peak_mb: float | None = None


@dataclass(slots=True)
class SimulationResult:
    """전체 simulation 요약."""

    initial_model_revision: str
    initial_prototype_version: str
    initial_validation: SimulationEvaluation
    final_validation: SimulationEvaluation
    rounds: tuple[SimulationRoundSummary, ...]
    client_evaluations: tuple[ClientEvaluationSummary, ...] = ()
    report_path: str | None = None


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
    """FL SSL method 선택과 report metadata 설정."""

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

    def require_shared_payload_matches_objective(
        self,
        objective_config: TrainingObjectiveConfig | None,
    ) -> None:
        """bootstrap state와 local update가 같은 backbone/LoRA snapshot을 쓰게 한다."""

        objective_backend_config = build_lora_classifier_training_backend_config(
            objective_config
        )
        mismatches: dict[str, object] = {}
        if self.backbone_payload() != objective_backend_config.to_backbone_payload():
            mismatches["backbone"] = {
                "round_runtime": self.backbone_payload(),
                "training_objective": (objective_backend_config.to_backbone_payload()),
            }
        if (
            self.lora_config_payload()
            != objective_backend_config.to_lora_config_payload()
        ):
            mismatches["lora_config"] = {
                "round_runtime": self.lora_config_payload(),
                "training_objective": (
                    objective_backend_config.to_lora_config_payload()
                ),
            }
        if mismatches:
            raise ValueError(
                "LoRA-classifier round_runtime.lora_classifier must match "
                f"training_task.objective shared payload config: {mismatches}."
            )


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
    ssl_method_config: FederatedSslMethodConfig
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
