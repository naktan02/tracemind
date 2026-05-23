"""LoRA-classifier local training backend facade."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from methods.adaptation.local_update_backend import AcceptedTrainingExample
from methods.adaptation.local_update_registry import (
    register_shared_adapter_training_backend,
)
from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
)
from methods.adaptation.lora_classifier.training.query_ssl_local_training import (
    LoraClassifierTrainerRuntimeConfig,
    QuerySslLoraClientTrainingResult,
    QuerySslLoraDeltaMaterializer,
    QuerySslLoraObjectiveRuntimeConfig,
    run_query_ssl_lora_classifier_training_core,
)
from methods.adaptation.lora_classifier.update.local_update import (
    LoraClassifierTrainExecutor,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_ADAPTER_KIND,
    LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    LoraClassifierDelta,
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
    LORA_CLASSIFIER_TRAINING_BACKEND_NAME,
    LoraClassifierTrainingBackendConfig,
    build_lora_classifier_training_backend_config,
)
from .update.payload_builder import build_lora_classifier_delta_update

LORA_CLASSIFIER_TRAINING_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name=LORA_CLASSIFIER_TRAINING_BACKEND_NAME,
    display_name=LORA_CLASSIFIER_TRAINING_BACKEND_NAME,
    implementation_module=("methods.adaptation.lora_classifier.training_backend"),
    core_method_name=LORA_CLASSIFIER_TRAINING_BACKEND_NAME,
    family_name=LORA_CLASSIFIER_ADAPTER_KIND,
    supported_adapter_kinds=(LORA_CLASSIFIER_ADAPTER_KIND,),
    accepted_payload_formats=(LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,),
    tags=("requires_raw_text", "artifact_ref_update"),
    metadata={
        "payload_format": LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
        "requires_raw_text": True,
        "produces_artifact_refs": True,
        "supports_live_stored_event_runtime": False,
    },
)


@dataclass(slots=True)
class LoraClassifierTrainingBackend:
    """raw text accepted example을 LoRA-classifier update payload로 바꾼다.

    이 backend는 raw text를 shared payload에 넣지 않는다. 현재는 실제 LoRA
    weight 파일을 생성하는 executor 없이 계약-compatible artifact ref를 남긴다.
    이후 PEFT 실행기는 `train_executor.py` seam 뒤에 연결한다.
    """

    backend_name: str = LORA_CLASSIFIER_TRAINING_BACKEND_NAME
    payload_format: str = LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT
    adapter_kind: str = LORA_CLASSIFIER_ADAPTER_KIND
    config: LoraClassifierTrainingBackendConfig = field(
        default_factory=LoraClassifierTrainingBackendConfig
    )
    train_executor: LoraClassifierTrainExecutor | None = None

    @classmethod
    def from_objective_config(
        cls,
        objective_config: TrainingObjectiveConfig | None,
    ) -> "LoraClassifierTrainingBackend":
        return cls(
            config=build_lora_classifier_training_backend_config(objective_config)
        )

    def build_update(
        self,
        *,
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        accepted_examples: tuple[AcceptedTrainingExample, ...],
        created_at: datetime,
    ) -> LoraClassifierDelta:
        return build_lora_classifier_delta_update(
            training_task=training_task,
            model_manifest=model_manifest,
            accepted_examples=accepted_examples,
            config=self.config,
            created_at=created_at,
            train_executor=self.train_executor,
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
        base_parameters: LoraClassifierMaterializedState,
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        query_ssl_config: QuerySslLoraObjectiveRuntimeConfig,
        trainer_runtime_config: LoraClassifierTrainerRuntimeConfig,
        created_at: datetime,
        delta_materializer: QuerySslLoraDeltaMaterializer,
    ) -> QuerySslLoraClientTrainingResult:
        """Query SSL raw rows를 methods-owned LoRA local core로 학습한다."""

        return run_query_ssl_lora_classifier_training_core(
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
        )

    def to_payload(self, update: SharedAdapterUpdate) -> SharedAdapterUpdatePayload:
        if not isinstance(update, LoraClassifierDelta):
            raise TypeError(
                "LoraClassifierTrainingBackend expects LoraClassifierDelta "
                f"for payload conversion, got {type(update)!r}."
            )
        return update

    def build_client_metrics(self, update: SharedAdapterUpdate) -> dict[str, float]:
        return build_lora_classifier_client_metrics(update)

    def matches_objective_config(
        self,
        objective_config: TrainingObjectiveConfig | None,
    ) -> bool:
        return self.config == build_lora_classifier_training_backend_config(
            objective_config
        )


def build_lora_classifier_client_metrics(
    update: SharedAdapterUpdate,
) -> dict[str, float]:
    if not isinstance(update, LoraClassifierDelta):
        raise TypeError(
            "LoraClassifierTrainingBackend expects LoraClassifierDelta "
            f"for metric extraction, got {type(update)!r}."
        )
    return {
        ClientMetricKeys.MEAN_CONFIDENCE: update.mean_confidence or 0.0,
        ClientMetricKeys.MEAN_MARGIN: update.mean_margin or 0.0,
        ClientMetricKeys.DELTA_L2_NORM: update.l2_norm(),
        "lora_training_rows": float(update.example_count),
        "lora_label_schema_size": float(len(update.label_schema)),
    }


@register_shared_adapter_training_backend(
    "lora_classifier_trainer",
    catalog_entry=LORA_CLASSIFIER_TRAINING_BACKEND_CATALOG_ENTRY,
)
def build_lora_classifier_training_backend(
    objective_config: TrainingObjectiveConfig | None,
) -> LoraClassifierTrainingBackend:
    """registry용 LoRA-classifier training backend factory."""

    return LoraClassifierTrainingBackend.from_objective_config(objective_config)
