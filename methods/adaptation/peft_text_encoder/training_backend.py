"""PEFT-backed classifier local training backend facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from methods.adaptation.local_update_backend import AcceptedTrainingExample
from methods.adaptation.local_update_registry import (
    register_shared_adapter_training_backend,
)
from methods.adaptation.peft_text_encoder.update.local_update import (
    PeftEncoderTrainExecutor,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_ADAPTER_KIND,
    PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    PeftClassifierDelta,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
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
    PEFT_CLASSIFIER_TRAINING_BACKEND_NAME,
    PeftEncoderTrainingBackendConfig,
    build_peft_classifier_training_backend_config,
)
from .training.query_ssl_local_training import (
    PeftEncoderTrainerRuntimeConfig,
    QuerySslPeftEncoderClientTrainingResult,
    QuerySslPeftEncoderDeltaMaterializer,
    QuerySslPeftEncoderObjectiveRuntimeConfig,
    run_query_ssl_peft_encoder_training_core,
)
from .update.payload_builder import build_peft_encoder_delta_update

PEFT_CLASSIFIER_TRAINING_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name=PEFT_CLASSIFIER_TRAINING_BACKEND_NAME,
    display_name=PEFT_CLASSIFIER_TRAINING_BACKEND_NAME,
    implementation_module=("methods.adaptation.peft_text_encoder.training_backend"),
    core_method_name=PEFT_CLASSIFIER_TRAINING_BACKEND_NAME,
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
    """raw text accepted example을 PEFT encoder classifier update payload로 바꾼다.

    이 backend는 raw text를 shared payload에 넣지 않는다. 현재는 실제 LoRA
    weight 파일을 생성하는 executor 없이 계약-compatible artifact ref를 남긴다.
    이후 PEFT 실행기는 `train_executor.py` seam 뒤에 연결한다.
    """

    backend_name: str = PEFT_CLASSIFIER_TRAINING_BACKEND_NAME
    payload_format: str = PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT
    adapter_kind: str = PEFT_CLASSIFIER_ADAPTER_KIND
    config: PeftEncoderTrainingBackendConfig = field(
        default_factory=lambda: build_peft_classifier_training_backend_config(None)
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
        """Query SSL raw rows를 methods-owned PEFT encoder local core로 학습한다."""

        return run_query_ssl_peft_encoder_training_core(
            client_id=client_id,
            seed=seed,
            labeled_rows=labeled_rows,
            unlabeled_rows=unlabeled_rows,
            diagnostic_unlabeled_rows=diagnostic_unlabeled_rows,
            labels=labels,
            base_parameters=base_parameters,
            training_task=training_task,
            model_manifest=model_manifest,
            query_ssl_config=query_ssl_config,
            lora_config=self.config,
            trainer_runtime_config=trainer_runtime_config,
            created_at=created_at,
            delta_materializer=delta_materializer,
            runtime_resource_cache=runtime_resource_cache,
            timing_recorder=timing_recorder,
            initial_query_ssl_algorithm_state=initial_query_ssl_algorithm_state,
        )

    def to_payload(self, update: SharedAdapterUpdate) -> SharedAdapterUpdatePayload:
        if not isinstance(update, PeftClassifierDelta):
            raise TypeError(
                "PeftEncoderTrainingBackend expects PEFT classifier delta "
                f"for payload conversion, got {type(update)!r}."
            )
        return update

    def build_client_metrics(self, update: SharedAdapterUpdate) -> dict[str, float]:
        return build_peft_encoder_client_metrics(update)

    def matches_objective_config(
        self,
        objective_config: TrainingObjectiveConfig | None,
    ) -> bool:
        return self.config == build_peft_classifier_training_backend_config(
            objective_config
        )


def build_peft_encoder_client_metrics(
    update: SharedAdapterUpdate,
) -> dict[str, float]:
    if not isinstance(update, PeftClassifierDelta):
        raise TypeError(
            "PeftEncoderTrainingBackend expects PEFT classifier delta "
            f"for metric extraction, got {type(update)!r}."
        )
    return {
        ClientMetricKeys.MEAN_CONFIDENCE: update.mean_confidence or 0.0,
        ClientMetricKeys.MEAN_MARGIN: update.mean_margin or 0.0,
        ClientMetricKeys.DELTA_L2_NORM: update.l2_norm(),
        "lora_training_rows": float(update.example_count),
        "lora_label_schema_size": float(len(update.label_schema)),
        "peft_classifier_training_rows": float(update.example_count),
        "peft_classifier_label_schema_size": float(len(update.label_schema)),
    }


@register_shared_adapter_training_backend(
    PEFT_CLASSIFIER_TRAINING_BACKEND_NAME,
    catalog_entry=PEFT_CLASSIFIER_TRAINING_BACKEND_CATALOG_ENTRY,
)
def build_peft_classifier_training_backend(
    objective_config: TrainingObjectiveConfig | None,
) -> PeftEncoderTrainingBackend:
    """registry용 PEFT-classifier v2 training backend factory."""

    return PeftEncoderTrainingBackend(
        backend_name=PEFT_CLASSIFIER_TRAINING_BACKEND_NAME,
        payload_format=PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
        adapter_kind=PEFT_CLASSIFIER_ADAPTER_KIND,
        config=build_peft_classifier_training_backend_config(objective_config),
    )
