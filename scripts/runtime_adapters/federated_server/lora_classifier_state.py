"""Simulation server-side LoRA-classifier state materialization bridge."""

from __future__ import annotations

from pathlib import Path

from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregationArtifactStore,
)
from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
    materialize_base_lora_classifier_state,
)
from methods.federated.aggregation.base import FederatedAggregationContext
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierState,
)


def materialize_simulation_lora_classifier_base_state(
    *,
    output_dir: Path,
    adapter_state: LoraClassifierState,
) -> LoraClassifierMaterializedState:
    """simulation output dir의 server artifact store에서 LoRA base state를 읽는다."""

    return materialize_base_lora_classifier_state(
        base_state=adapter_state,
        context=FederatedAggregationContext(
            next_model_revision=adapter_state.model_revision,
            aggregated_at=adapter_state.updated_at,
            artifact_loader=AggregationArtifactStore(
                state_root=output_dir / "main_server" / "aggregation_artifacts"
            ),
        ),
    )
