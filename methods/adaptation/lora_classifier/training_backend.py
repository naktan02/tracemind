"""LoRA-classifier local training backend facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from methods.adaptation.local_update_backend import AcceptedTrainingExample
from methods.adaptation.local_update_registry import (
    register_shared_adapter_training_backend,
)
from methods.adaptation.lora_classifier.local_update import (
    LoraClassifierTrainExecutor,
)
from shared.src.contracts.adapter_contracts import (
    LoraClassifierDelta,
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_family_metadata import LORA_CLASSIFIER_FAMILY_METADATA
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.registry_catalog_metadata import (
    RegistryCatalogEntry,
)
from shared.src.contracts.training_contracts import (
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
from .metrics import build_lora_classifier_client_metrics
from .payload_builder import build_lora_classifier_delta_update

LORA_CLASSIFIER_TRAINING_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name=LORA_CLASSIFIER_TRAINING_BACKEND_NAME,
    display_name=LORA_CLASSIFIER_TRAINING_BACKEND_NAME,
    implementation_module=(
        "methods.adaptation.lora_classifier.training_backend"
    ),
    core_method_name=LORA_CLASSIFIER_TRAINING_BACKEND_NAME,
    family_name=LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind,
    supported_adapter_kinds=(LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind,),
    accepted_payload_formats=(
        LORA_CLASSIFIER_FAMILY_METADATA.canonical_update_payload_format,
    ),
    tags=("requires_raw_text", "artifact_ref_update"),
    metadata={
        "payload_format": (
            LORA_CLASSIFIER_FAMILY_METADATA.canonical_update_payload_format
        ),
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
    payload_format: str = (
        LORA_CLASSIFIER_FAMILY_METADATA.canonical_update_payload_format
    )
    adapter_kind: str = LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind
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


@register_shared_adapter_training_backend(
    "lora_classifier_trainer",
    catalog_entry=LORA_CLASSIFIER_TRAINING_BACKEND_CATALOG_ENTRY,
)
def build_lora_classifier_training_backend(
    objective_config: TrainingObjectiveConfig | None,
) -> LoraClassifierTrainingBackend:
    """registry용 LoRA-classifier training backend factory."""

    return LoraClassifierTrainingBackend.from_objective_config(objective_config)
