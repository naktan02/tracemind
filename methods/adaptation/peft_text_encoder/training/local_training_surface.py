"""PEFT Query SSL local training request surface.

이 모듈은 central/offline, FL simulation, live agent가 같은 methods-owned
local training core를 호출할 때 공유하는 입력 표면을 소유한다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask

if TYPE_CHECKING:
    from .query_ssl_federated_update import QuerySslPeftEncoderDeltaMaterializer


class QuerySslPeftEncoderObjectiveRuntimeConfig(Protocol):
    """Query SSL PEFT encoder local core가 필요한 objective config surface."""

    algorithm_name: str
    parameters: Mapping[str, object]
    strong_view_policy: str
    unlabeled_batch_size: int | None
    drop_last_train_batches: bool
    drop_last_unlabeled_batches: bool


class PeftEncoderTrainerRuntimeConfig(Protocol):
    """PEFT text encoder/head 모델 로딩/학습 core가 필요한 runtime config surface."""

    device: str
    classifier_dropout: float
    cache_dir: str | None
    local_files_only: bool
    trust_remote_code: bool


@dataclass(frozen=True, slots=True)
class QuerySslPeftEncoderLocalTrainerOptions:
    """문맥별 adapter가 공통 local session에 넘기는 trainer 실행 옵션."""

    classifier_learning_rate: float | None = None
    weight_decay: float = 0.0
    log_every_steps: int = 0
    resume_checkpoint_path: str | Path | None = None
    resume_checkpoint_output_dir: str | Path | None = None
    resume_checkpoint_every_epochs: int = 0


@dataclass(frozen=True, slots=True)
class QuerySslPeftEncoderLocalSessionRequest:
    """central/agent/simulation이 공유하는 PEFT Query SSL local 학습 요청."""

    seed: int
    labeled_rows: Sequence[LabeledQueryRow]
    unlabeled_rows: Sequence[LabeledQueryRow]
    labels: Sequence[str]
    base_parameters: PeftEncoderMaterializedState
    training_task: TrainingTask
    query_ssl_config: QuerySslPeftEncoderObjectiveRuntimeConfig
    peft_config: PeftEncoderTrainingBackendConfig
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig
    diagnostic_unlabeled_rows: Sequence[LabeledQueryRow] | None = None
    selection_rows: Sequence[LabeledQueryRow] | None = None
    runtime_resource_cache: RuntimeResourceCache | None = None
    timing_recorder: TimingRecorder | None = None
    initial_query_ssl_algorithm_state: Mapping[str, object] | None = None
    trainer_options: QuerySslPeftEncoderLocalTrainerOptions | None = None


@dataclass(frozen=True, slots=True)
class QuerySslPeftEncoderUpdateRequest:
    """federated/live update 생성을 포함하는 PEFT Query SSL 요청."""

    client_id: str
    local_session: QuerySslPeftEncoderLocalSessionRequest
    model_manifest: ModelManifest
    created_at: datetime
    delta_materializer: "QuerySslPeftEncoderDeltaMaterializer"
