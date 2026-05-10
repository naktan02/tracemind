"""Diagonal-scale heuristic training backend."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from methods.adaptation.diagonal_scale.config import (
    DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG,
    DiagonalScaleHeuristicTrainingBackendConfig,
)
from methods.adaptation.diagonal_scale.heuristic_update import (
    build_diagonal_scale_client_metrics,
    build_diagonal_scale_heuristic_config,
    build_diagonal_scale_heuristic_update,
)
from methods.adaptation.local_update_backend import AcceptedTrainingExample
from methods.adaptation.local_update_registry import (
    register_shared_adapter_training_backend,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    DIAGONAL_SCALE_ADAPTER_KIND,
    DIAGONAL_SCALE_UPDATE_PAYLOAD_FORMAT,
    VectorAdapterDelta,
)
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

DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="diagonal_scale_heuristic",
    display_name="diagonal_scale_heuristic",
    implementation_module=("methods.adaptation.diagonal_scale.training_backend"),
    core_method_name="diagonal_scale_heuristic",
    family_name=DIAGONAL_SCALE_ADAPTER_KIND,
    supported_adapter_kinds=(DIAGONAL_SCALE_ADAPTER_KIND,),
    accepted_payload_formats=(DIAGONAL_SCALE_UPDATE_PAYLOAD_FORMAT,),
    metadata={"payload_format": DIAGONAL_SCALE_UPDATE_PAYLOAD_FORMAT},
)


@dataclass(slots=True)
class DiagonalScaleHeuristicTrainingBackend:
    """결정적 통계 기반으로 diagonal scale adapter update를 만든다."""

    backend_name: str = "diagonal_scale_heuristic"
    payload_format: str = DIAGONAL_SCALE_UPDATE_PAYLOAD_FORMAT
    adapter_kind: str = DIAGONAL_SCALE_ADAPTER_KIND
    config: DiagonalScaleHeuristicTrainingBackendConfig = (
        DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG
    )

    @classmethod
    def from_objective_config(
        cls,
        objective_config: TrainingObjectiveConfig | None,
    ) -> "DiagonalScaleHeuristicTrainingBackend":
        return cls(config=build_diagonal_scale_heuristic_config(objective_config))

    def build_update(
        self,
        *,
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        accepted_examples: tuple[AcceptedTrainingExample, ...],
        created_at: datetime,
    ) -> VectorAdapterDelta:
        return build_diagonal_scale_heuristic_update(
            training_task=training_task,
            model_manifest=model_manifest,
            accepted_examples=accepted_examples,
            created_at=created_at,
            config=self.config,
            adapter_kind=self.adapter_kind,
        )

    def to_payload(self, update: SharedAdapterUpdate) -> SharedAdapterUpdatePayload:
        if not isinstance(update, VectorAdapterDelta):
            raise TypeError(
                "DiagonalScaleHeuristicTrainingBackend expects VectorAdapterDelta "
                f"for payload conversion, got {type(update)!r}."
            )
        return update

    def build_client_metrics(self, update: SharedAdapterUpdate) -> dict[str, float]:
        if not isinstance(update, VectorAdapterDelta):
            raise TypeError(
                "DiagonalScaleHeuristicTrainingBackend expects VectorAdapterDelta "
                f"for metric extraction, got {type(update)!r}."
            )
        return build_diagonal_scale_client_metrics(update)

    def matches_objective_config(
        self,
        objective_config: TrainingObjectiveConfig | None,
    ) -> bool:
        return self.config == build_diagonal_scale_heuristic_config(objective_config)


@register_shared_adapter_training_backend(
    "diagonal_scale_heuristic",
    "synthetic_vector_adapter",
    catalog_entry=DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CATALOG_ENTRY,
)
def build_diagonal_scale_heuristic_training_backend(
    objective_config: TrainingObjectiveConfig | None,
) -> DiagonalScaleHeuristicTrainingBackend:
    """registry용 diagonal-scale heuristic backend factory."""

    return DiagonalScaleHeuristicTrainingBackend.from_objective_config(objective_config)
