"""LoRA-classifier simulation base state materialization bridge."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregationArtifactStore,
)
from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
    materialize_base_lora_classifier_state,
)
from methods.federated.aggregation.base import FederatedAggregationContext
from scripts.experiments.fl_ssl.federated_simulation.runtime_resources import (
    RoundBaseSnapshotCache,
    RoundBaseSnapshotCacheKey,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierState,
)

LORA_CLASSIFIER_BASE_STATE_MATERIALIZER_NAME = "lora_classifier_base_state.v1"


def load_lora_classifier_base_parameters(
    *,
    active_adapter_state: LoraClassifierState,
    output_dir: Path,
    aggregated_at: datetime,
    round_base_snapshot_cache: RoundBaseSnapshotCache | None = None,
) -> LoraClassifierMaterializedState:
    """server-published LoRA-classifier base state를 materialize한다."""

    if round_base_snapshot_cache is None:
        return _materialize_base_parameters(
            active_adapter_state=active_adapter_state,
            output_dir=output_dir,
            aggregated_at=aggregated_at,
        )
    snapshot = round_base_snapshot_cache.get_or_materialize(
        key=_cache_key(active_adapter_state),
        materialize=lambda: _materialize_base_parameters(
            active_adapter_state=active_adapter_state,
            output_dir=output_dir,
            aggregated_at=aggregated_at,
        ),
    )
    if not isinstance(snapshot, LoraClassifierMaterializedState):
        raise TypeError(
            "Round base snapshot cache returned unexpected LoRA-classifier state: "
            f"{type(snapshot)!r}."
        )
    return snapshot


def _materialize_base_parameters(
    *,
    active_adapter_state: LoraClassifierState,
    output_dir: Path,
    aggregated_at: datetime,
) -> LoraClassifierMaterializedState:
    return materialize_base_lora_classifier_state(
        base_state=active_adapter_state,
        context=FederatedAggregationContext(
            next_model_revision=active_adapter_state.model_revision,
            aggregated_at=aggregated_at,
            artifact_loader=AggregationArtifactStore(
                state_root=output_dir / "main_server" / "aggregation_artifacts"
            ),
        ),
    )


def _cache_key(active_adapter_state: LoraClassifierState) -> RoundBaseSnapshotCacheKey:
    return RoundBaseSnapshotCacheKey(
        adapter_kind=str(active_adapter_state.adapter_kind),
        model_revision=str(active_adapter_state.model_revision),
        schema_version=str(active_adapter_state.schema_version),
        artifact_refs=(
            (
                "lora_adapter_artifact_ref",
                _ref_value(active_adapter_state.lora_adapter_artifact_ref),
            ),
            (
                "classifier_head_artifact_ref",
                _ref_value(active_adapter_state.classifier_head_artifact_ref),
            ),
        ),
        materializer_name=LORA_CLASSIFIER_BASE_STATE_MATERIALIZER_NAME,
    )


def _ref_value(value: str | None) -> str:
    return "" if value is None else cast(str, value)
