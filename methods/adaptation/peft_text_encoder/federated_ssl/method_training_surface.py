"""Method-owned FSSL PEFT local training request surface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.training import (
    query_ssl_local_training as qssl_training,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder
from methods.federated_ssl.hooks.peer_context import FederatedSslPeerContext
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask

PeftEncoderTrainerRuntimeConfig = qssl_training.PeftEncoderTrainerRuntimeConfig
QuerySslPeftEncoderDeltaMaterializer = (
    qssl_training.QuerySslPeftEncoderDeltaMaterializer
)
QuerySslPeftEncoderObjectiveRuntimeConfig = (
    qssl_training.QuerySslPeftEncoderObjectiveRuntimeConfig
)


class FederatedSslMethodLocalTrainingConfig(Protocol):
    """method-owned PEFT local core가 필요한 method config surface."""

    name: str
    scenario: str | None
    effective_parameters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FsslPeftEncoderMethodTrainingRequest:
    """agent/simulation이 공유하는 method-owned PEFT FSSL local 학습 요청."""

    client_id: str
    seed: int
    labeled_rows: Sequence[LabeledQueryRow]
    unlabeled_rows: Sequence[LabeledQueryRow]
    labels: Sequence[str]
    base_parameters: PeftEncoderMaterializedState
    training_task: TrainingTask
    model_manifest: ModelManifest
    ssl_method_config: FederatedSslMethodLocalTrainingConfig
    local_ssl_policy_name: str
    query_ssl_config: QuerySslPeftEncoderObjectiveRuntimeConfig | None
    strong_view_policy: str
    unlabeled_batch_size: int | None
    peft_config: PeftEncoderTrainingBackendConfig
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig
    created_at: datetime
    delta_materializer: QuerySslPeftEncoderDeltaMaterializer
    diagnostic_unlabeled_rows: Sequence[LabeledQueryRow] | None = None
    base_partition_parameters: Mapping[str, PeftEncoderMaterializedState] | None = None
    previous_client_partition_parameters: (
        Mapping[str, PeftEncoderMaterializedState] | None
    ) = None
    peer_context: FederatedSslPeerContext | None = None
    helper_weak_probability_provider: object | None = None
    peer_probe_rows: Sequence[LabeledQueryRow] | None = None
    runtime_resource_cache: RuntimeResourceCache | None = None
    timing_recorder: TimingRecorder | None = None
    initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None
