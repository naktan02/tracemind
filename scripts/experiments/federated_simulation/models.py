"""Federated simulation용 설정/결과 모델."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)


@dataclass(slots=True)
class FederatedClientShard:
    """한 client에 할당된 train row 묶음."""

    client_id: str
    rows: list[dict[str, Any]]


@dataclass(slots=True)
class FederatedDatasetSplit:
    """bootstrap과 client shard로 나눈 train subset."""

    bootstrap_rows: list[dict[str, Any]]
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


@dataclass(slots=True)
class FederatedShardPolicyConfig:
    """client shard 분할 정책 설정."""

    name: str
    dominant_ratio: float
    client_id_prefix: str


@dataclass(slots=True)
class FederatedTrainingTaskConfig:
    """round별 로컬 학습 task 설정."""

    local_epochs: int
    batch_size: int
    learning_rate: float
    max_steps: int
    min_required_examples: int
    objective_config: TrainingObjectiveConfig
    selection_policy: TrainingSelectionPolicy
    gradient_clip_norm: float | None = None


@dataclass(slots=True)
class FederatedValidationConfig:
    """validation score/acceptance 계산 설정."""

    similarity_name: str
    score_policy_name: str
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
