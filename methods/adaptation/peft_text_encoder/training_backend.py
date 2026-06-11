"""PEFT text encoder local training backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from methods.adaptation.local_update_backend import AcceptedTrainingExample
from methods.adaptation.local_update_registry import (
    register_shared_adapter_training_backend,
)
from methods.adaptation.peft_text_encoder.update.local_update import (
    PeftEncoderTrainExecutor,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_ADAPTER_KIND,
    PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    PeftClassifierDelta,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.registry_catalog_metadata import (
    RegistryCatalogEntry,
)
from shared.src.contracts.training_contracts import (
    ClientMetricKeys,
    TrainingObjectiveConfig,
    TrainingTask,
)
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from .config import (
    PEFT_ENCODER_TRAINING_BACKEND_NAME,
    PeftEncoderTrainingBackendConfig,
    build_peft_encoder_training_backend_config,
)
from .training.local_training_surface import (
    QuerySslPeftEncoderLocalSessionRequest,
    QuerySslPeftEncoderUpdateRequest,
)
from .training.query_ssl_local_training import (
    QuerySslPeftEncoderClientTrainingResult,
    run_query_ssl_peft_encoder_update,
)
from .update.payload_builder import build_peft_encoder_delta_update

PEFT_ENCODER_TRAINING_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name=PEFT_ENCODER_TRAINING_BACKEND_NAME,
    display_name=PEFT_ENCODER_TRAINING_BACKEND_NAME,
    implementation_module=("methods.adaptation.peft_text_encoder.training_backend"),
    core_method_name=PEFT_ENCODER_TRAINING_BACKEND_NAME,
    family_name=PEFT_CLASSIFIER_ADAPTER_KIND,
    supported_adapter_kinds=(PEFT_CLASSIFIER_ADAPTER_KIND,),
    accepted_payload_formats=(PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,),
    tags=("requires_raw_text", "artifact_ref_update"),
    metadata={
        "payload_format": PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
        "requires_raw_text": True,
        "produces_artifact_refs": True,
        "supports_live_stored_event_runtime": False,
    },
)


@dataclass(slots=True)
class PeftEncoderTrainingBackend:
    """raw text accepted example을 PEFT encoder update payload로 바꾼다.

    이 backend는 raw text를 shared payload에 넣지 않는다. 현재는 실제 PEFT
    adapter weight 파일을 생성하는 executor 없이 계약-compatible artifact ref를 남긴다.
    이후 PEFT 실행기는 `train_executor.py` seam 뒤에 연결한다.
    """

    backend_name: str = PEFT_ENCODER_TRAINING_BACKEND_NAME
    payload_format: str = PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT
    adapter_kind: str = PEFT_CLASSIFIER_ADAPTER_KIND
    config: PeftEncoderTrainingBackendConfig = field(
        default_factory=lambda: build_peft_encoder_training_backend_config(None)
    )
    train_executor: PeftEncoderTrainExecutor | None = None

    def build_update(
        self,
        *,
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        accepted_examples: tuple[AcceptedTrainingExample, ...],
        created_at: datetime,
    ) -> PeftClassifierDelta:
        return build_peft_encoder_delta_update(
            training_task=training_task,
            model_manifest=model_manifest,
            accepted_examples=accepted_examples,
            config=self.config,
            created_at=created_at,
            train_executor=self.train_executor,
        )

    def with_simulation_inline_train_executor(self) -> "PeftEncoderTrainingBackend":
        """simulation inline delta 실행기가 필요한 경우 executor를 붙인다."""

        from methods.adaptation.peft_text_encoder.config import (
            PEFT_ENCODER_DELTA_FORMAT_INLINE,
        )
        from methods.adaptation.peft_text_encoder.update import (
            simulation_inline_delta,
        )

        if self.config.delta_format != PEFT_ENCODER_DELTA_FORMAT_INLINE:
            return self
        return PeftEncoderTrainingBackend(
            backend_name=self.backend_name,
            payload_format=self.payload_format,
            adapter_kind=self.adapter_kind,
            config=self.config,
            train_executor=(
                simulation_inline_delta.SimulationInlinePeftEncoderTrainExecutor()
            ),
        )

    def build_query_ssl_update(
        self,
        request: QuerySslPeftEncoderUpdateRequest,
    ) -> QuerySslPeftEncoderClientTrainingResult:
        """Query SSL raw rows를 methods-owned PEFT encoder local core로 학습한다."""

        session = request.local_session
        return run_query_ssl_peft_encoder_update(
            QuerySslPeftEncoderUpdateRequest(
                client_id=request.client_id,
                local_session=QuerySslPeftEncoderLocalSessionRequest(
                    seed=session.seed,
                    labeled_rows=session.labeled_rows,
                    unlabeled_rows=session.unlabeled_rows,
                    diagnostic_unlabeled_rows=session.diagnostic_unlabeled_rows,
                    selection_rows=session.selection_rows,
                    labels=session.labels,
                    base_parameters=session.base_parameters,
                    training_task=session.training_task,
                    query_ssl_config=session.query_ssl_config,
                    peft_config=self.config,
                    trainer_runtime_config=session.trainer_runtime_config,
                    runtime_resource_cache=session.runtime_resource_cache,
                    timing_recorder=session.timing_recorder,
                    initial_query_ssl_algorithm_state=(
                        session.initial_query_ssl_algorithm_state
                    ),
                    trainer_options=session.trainer_options,
                ),
                model_manifest=request.model_manifest,
                created_at=request.created_at,
                delta_materializer=request.delta_materializer,
            )
        )

    def to_payload(self, update: SharedAdapterUpdate) -> SharedAdapterUpdatePayload:
        if not isinstance(update, PeftClassifierDelta):
            raise TypeError(
                "PeftEncoderTrainingBackend expects PEFT text encoder delta "
                f"for payload conversion, got {type(update)!r}."
            )
        return update

    def build_client_metrics(self, update: SharedAdapterUpdate) -> dict[str, float]:
        return build_peft_encoder_client_metrics(update)

    def matches_objective_config(
        self,
        objective_config: TrainingObjectiveConfig | None,
    ) -> bool:
        return self.config == build_peft_encoder_training_backend_config(
            objective_config
        )


def build_peft_encoder_client_metrics(
    update: SharedAdapterUpdate,
) -> dict[str, float]:
    if not isinstance(update, PeftClassifierDelta):
        raise TypeError(
            "PeftEncoderTrainingBackend expects PEFT text encoder delta "
            f"for metric extraction, got {type(update)!r}."
        )
    return {
        ClientMetricKeys.MEAN_CONFIDENCE: update.mean_confidence or 0.0,
        ClientMetricKeys.MEAN_MARGIN: update.mean_margin or 0.0,
        ClientMetricKeys.DELTA_L2_NORM: update.l2_norm(),
        ClientMetricKeys.SELECTED_EXAMPLES: float(update.example_count),
        "label_schema_size": float(len(update.label_schema)),
    }


@register_shared_adapter_training_backend(
    PEFT_ENCODER_TRAINING_BACKEND_NAME,
    catalog_entry=PEFT_ENCODER_TRAINING_BACKEND_CATALOG_ENTRY,
)
def build_peft_encoder_training_backend(
    objective_config: TrainingObjectiveConfig | None,
) -> PeftEncoderTrainingBackend:
    """registry용 PEFT text encoder training backend factory."""

    return PeftEncoderTrainingBackend(
        backend_name=PEFT_ENCODER_TRAINING_BACKEND_NAME,
        payload_format=PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
        adapter_kind=PEFT_CLASSIFIER_ADAPTER_KIND,
        config=build_peft_encoder_training_backend_config(objective_config),
    )
