"""중앙 Query SSL text encoder session 공통 요청 표면."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.training_contracts import TrainingTask


class QuerySslTextEncoderObjectiveRuntimeConfig(Protocol):
    """Text encoder Query SSL core가 필요한 objective config surface."""

    algorithm_name: str
    parameters: Mapping[str, object]
    strong_view_policy: str
    unlabeled_batch_size: int | None
    drop_last_train_batches: bool
    drop_last_unlabeled_batches: bool


class TextEncoderTrainerRuntimeConfig(Protocol):
    """Text encoder/head 모델 로딩/학습 core가 필요한 runtime config surface."""

    device: str
    classifier_dropout: float
    cache_dir: str | None
    local_files_only: bool
    trust_remote_code: bool


@dataclass(frozen=True, slots=True)
class QuerySslTextEncoderLocalTrainerOptions:
    """중앙 runner가 surface local session에 넘기는 trainer 실행 옵션."""

    classifier_learning_rate: float | None = None
    weight_decay: float = 0.0
    log_every_steps: int = 0
    resume_checkpoint_path: str | Path | None = None
    resume_checkpoint_output_dir: str | Path | None = None
    resume_checkpoint_every_epochs: int = 0


@dataclass(frozen=True, slots=True)
class CentralQuerySslTextEncoderSessionRequest:
    """중앙 pooled SSL cfg/context를 surface-neutral local session 입력으로 묶는다."""

    cfg: Any
    seed: int
    labeled_rows: Sequence[LabeledQueryRow]
    unlabeled_rows: Sequence[LabeledQueryRow]
    labels: Sequence[str]
    model: Any
    tokenizer: Any
    training_task: TrainingTask
    query_ssl_config: QuerySslTextEncoderObjectiveRuntimeConfig
    trainer_runtime_config: TextEncoderTrainerRuntimeConfig
    diagnostic_unlabeled_rows: Sequence[LabeledQueryRow] | None = None
    selection_rows: Sequence[LabeledQueryRow] | None = None
    initial_query_ssl_algorithm_state: Mapping[str, object] | None = None
    trainer_options: QuerySslTextEncoderLocalTrainerOptions | None = None
