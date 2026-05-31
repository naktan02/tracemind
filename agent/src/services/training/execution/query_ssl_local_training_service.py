"""Query SSL raw-row local training execution service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Protocol, cast

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from methods.adaptation.local_update_backend import SharedAdapterTrainingBackend
from methods.adaptation.local_update_registry import (
    build_shared_adapter_training_backend,
)
from methods.adaptation.peft_text_encoder.training import (
    query_ssl_local_training as qssl_training,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask
from shared.src.domain.services.clock import Clock, SystemUtcClock
from shared.src.services.secure_update_codec import (
    NoOpSecureUpdateCodec,
    SecureUpdateCodec,
)

PeftEncoderTrainerRuntimeConfig = qssl_training.PeftEncoderTrainerRuntimeConfig
QuerySslPeftEncoderClientTrainingResult = (
    qssl_training.QuerySslPeftEncoderClientTrainingResult
)
QuerySslPeftEncoderDeltaMaterializer = (
    qssl_training.QuerySslPeftEncoderDeltaMaterializer
)
QuerySslPeftEncoderObjectiveRuntimeConfig = (
    qssl_training.QuerySslPeftEncoderObjectiveRuntimeConfig
)


@dataclass(slots=True)
class QuerySslPeftEncoderLocalTrainingRequest:
    """Query SSL PEFT encoder local training 실행 입력 묶음."""

    client_id: str
    seed: int
    labeled_rows: Sequence[LabeledQueryRow]
    unlabeled_rows: Sequence[LabeledQueryRow]
    labels: Sequence[str]
    base_parameters: PeftEncoderMaterializedState
    training_task: TrainingTask
    model_manifest: ModelManifest
    query_ssl_config: QuerySslPeftEncoderObjectiveRuntimeConfig
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig
    delta_materializer: QuerySslPeftEncoderDeltaMaterializer
    created_at: datetime | None = None
    agent_id: str | None = None
    diagnostic_unlabeled_rows: Sequence[LabeledQueryRow] | None = None
    runtime_resource_cache: RuntimeResourceCache | None = None
    timing_recorder: TimingRecorder | None = None
    persist_update_artifact: bool = True
    initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None


class QuerySslPeftEncoderTrainingBackend(Protocol):
    """Query SSL PEFT encoder raw-row local training capability."""

    backend_name: str

    def matches_objective_config(self, objective_config: object | None) -> bool:
        """현재 backend 인스턴스가 objective config와 일치하는지 판단한다."""

    def build_query_ssl_update(
        self,
        *,
        client_id: str,
        seed: int,
        labeled_rows: Sequence[LabeledQueryRow],
        unlabeled_rows: Sequence[LabeledQueryRow],
        diagnostic_unlabeled_rows: Sequence[LabeledQueryRow] | None,
        labels: Sequence[str],
        base_parameters: PeftEncoderMaterializedState,
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        query_ssl_config: QuerySslPeftEncoderObjectiveRuntimeConfig,
        trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
        created_at: datetime,
        delta_materializer: QuerySslPeftEncoderDeltaMaterializer,
        runtime_resource_cache: RuntimeResourceCache | None = None,
        timing_recorder: TimingRecorder | None = None,
        initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
    ) -> QuerySslPeftEncoderClientTrainingResult:
        """Query SSL raw rows를 학습해 local update payload를 만든다."""


@dataclass(slots=True)
class QuerySslLocalTrainingService:
    """Query SSL raw-row local training을 backend capability로 실행한다."""

    repository: TrainingArtifactRepository = field(
        default_factory=TrainingArtifactRepository
    )
    backend: SharedAdapterTrainingBackend | None = None
    secure_update_codec: SecureUpdateCodec = field(
        default_factory=NoOpSecureUpdateCodec
    )
    clock: Clock = field(default_factory=SystemUtcClock)

    def run_peft_encoder(
        self,
        request: QuerySslPeftEncoderLocalTrainingRequest,
    ) -> QuerySslPeftEncoderClientTrainingResult:
        """Query SSL PEFT encoder raw-row training을 선택된 local backend로 실행한다."""

        if (
            request.training_task.model_revision
            != request.model_manifest.model_revision
        ):
            raise ValueError("TrainingTask model_revision must match ModelManifest.")

        backend = self._resolve_peft_encoder_backend(
            training_task=request.training_task
        )
        effective_created_at = request.created_at or self.clock.now()
        result = backend.build_query_ssl_update(
            client_id=request.client_id,
            seed=request.seed,
            labeled_rows=request.labeled_rows,
            unlabeled_rows=request.unlabeled_rows,
            diagnostic_unlabeled_rows=request.diagnostic_unlabeled_rows,
            labels=request.labels,
            base_parameters=request.base_parameters,
            training_task=request.training_task,
            model_manifest=request.model_manifest,
            query_ssl_config=request.query_ssl_config,
            trainer_runtime_config=request.trainer_runtime_config,
            created_at=effective_created_at,
            delta_materializer=request.delta_materializer,
            runtime_resource_cache=request.runtime_resource_cache,
            timing_recorder=request.timing_recorder,
            initial_query_ssl_algorithm_state=(
                request.initial_query_ssl_algorithm_state
            ),
        )
        if request.persist_update_artifact:
            if request.timing_recorder is None:
                self.repository.save_shared_adapter_update(
                    result.update_envelope.update_id,
                    result.update_payload,
                )
            else:
                with request.timing_recorder.measure("agent_repository_save_seconds"):
                    self.repository.save_shared_adapter_update(
                        result.update_envelope.update_id,
                        result.update_payload,
                    )
        update_envelope = result.update_envelope
        if request.agent_id is not None:
            update_envelope = update_envelope.model_copy(
                update={"agent_id": request.agent_id}
            )
        encoded_envelope = self.secure_update_codec.encode_for_submission(
            envelope=update_envelope,
            training_task=request.training_task,
        )
        return replace(result, update_envelope=encoded_envelope)

    def _resolve_peft_encoder_backend(
        self,
        *,
        training_task: TrainingTask,
    ) -> QuerySslPeftEncoderTrainingBackend:
        backend = self._resolve_backend(training_task=training_task)
        if not hasattr(backend, "build_query_ssl_update"):
            raise ValueError(
                "Selected local training backend does not support Query SSL PEFT "
                f"encoder: {backend.backend_name}."
            )
        return cast(QuerySslPeftEncoderTrainingBackend, backend)

    def _resolve_backend(
        self,
        *,
        training_task: TrainingTask,
    ) -> SharedAdapterTrainingBackend:
        backend_name = training_task.objective_config.training_backend_name
        if self.backend is not None and backend_name == self.backend.backend_name:
            if self.backend.matches_objective_config(training_task.objective_config):
                return self.backend
        return build_shared_adapter_training_backend(
            backend_name,
            objective_config=training_task.objective_config,
        )


def run_query_ssl_peft_encoder_local_training(
    *,
    local_training_service: QuerySslLocalTrainingService,
    request: QuerySslPeftEncoderLocalTrainingRequest,
) -> QuerySslPeftEncoderClientTrainingResult:
    """config-declared bridge가 PEFT encoder Query SSL service를 실행하는 entrypoint."""

    return local_training_service.run_peft_encoder(request)
