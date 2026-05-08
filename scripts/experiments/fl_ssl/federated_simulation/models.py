"""Federated simulation용 설정/결과 모델."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.federated_ssl.experiment_profile import FederatedSslExperimentProfile
from methods.federated_ssl.local_update_profile import LocalUpdateProfile
from methods.prototype.building.base import PrototypeBuildStrategy
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec


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


@dataclass(slots=True)
class SimulationEvaluation:
    """validation 평가 결과."""

    row_count: int
    top1_accuracy: float
    accepted_ratio: float
    macro_f1: float = 0.0
    expected_calibration_error: float = 0.0
    per_label: dict[str, dict[str, int | float]] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)


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


class FederatedTrainingTaskConfig(Protocol):
    """federated simulation이 요구하는 round task config surface."""

    local_epochs: int
    batch_size: int
    learning_rate: float
    max_steps: int
    min_required_examples: int
    gradient_clip_norm: float | None
    objective_config: TrainingObjectiveConfig
    selection_policy: TrainingSelectionPolicy

    def to_round_open_request(
        self,
        *,
        active_manifest: ModelManifest,
        round_id: str,
    ) -> Any:
        """main_server round open request로 변환한다."""


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

    backbone_model_id: str
    backbone_revision: str
    tokenizer_model_id: str
    tokenizer_revision: str
    pooling: str
    max_length: int
    task_prefix: str
    peft_adapter_name: str
    rank: int
    alpha: int
    dropout: float
    bias: str
    target_modules: str
    use_rslora: bool
    artifact_format: str = "simulation_lora_classifier_state_ref"
    lora_adapter_artifact_ref: str | None = None
    classifier_head_artifact_ref: str | None = None

    def backbone_payload(self) -> dict[str, str | int]:
        """shared lora_classifier state에 넣을 backbone/tokenizer snapshot."""

        return {
            "backbone_model_id": self.backbone_model_id,
            "backbone_revision": self.backbone_revision,
            "tokenizer_model_id": self.tokenizer_model_id,
            "tokenizer_revision": self.tokenizer_revision,
            "pooling": self.pooling,
            "max_length": self.max_length,
            "task_prefix": self.task_prefix,
        }

    def lora_config_payload(self) -> dict[str, str | int | float | bool]:
        """shared lora_classifier state에 넣을 LoRA config snapshot."""

        return {
            "peft_adapter_name": self.peft_adapter_name,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "bias": self.bias,
            "target_modules": self.target_modules,
            "use_rslora": self.use_rslora,
        }


@dataclass(slots=True)
class FederatedRoundRuntimeConfig:
    """simulation이 사용할 shared family / aggregation backend 설정."""

    adapter_family_name: str
    aggregation_backend_name: str
    classifier_head_bootstrap_logit_scale: float = 8.0
    lora_classifier: FederatedLoraClassifierRuntimeConfig | None = None
    profile_name: str | None = None


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
    report_config: FederatedReportConfig | None = None
    local_update_profile: LocalUpdateProfile | None = None
    experiment_profile: FederatedSslExperimentProfile | None = None
