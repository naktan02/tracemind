"""Federated-agent simulation base state materialization bridge."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregationArtifactStore,
)
from methods.adaptation.text_classifier.peft_encoder.update.base_state_snapshot import (
    PEFT_ENCODER_BASE_STATE_MATERIALIZER_NAME,
    peft_encoder_base_state_artifact_refs,
)
from methods.adaptation.text_classifier.peft_encoder.update.materialization import (
    LoraClassifierMaterializedState,
    materialize_base_peft_encoder_partitioned_state,
    materialize_base_peft_encoder_state,
)
from methods.federated.aggregation.base import FederatedAggregationContext
from scripts.experiments.fl_ssl.federated_simulation.runtime_resources import (
    RoundBaseSnapshotCache,
    RoundBaseSnapshotCacheKey,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierState,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierState,
)

PeftEncoderState = LoraClassifierState | PeftClassifierState

PEFT_ENCODER_BASE_PARTITIONED_STATE_MATERIALIZER_NAME = (
    "peft_encoder_base_partitioned_state_v1"
)
LORA_CLASSIFIER_BASE_PARTITIONED_STATE_MATERIALIZER_NAME = (
    PEFT_ENCODER_BASE_PARTITIONED_STATE_MATERIALIZER_NAME
)


def load_peft_encoder_base_parameters(
    *,
    active_adapter_state: PeftEncoderState,
    output_dir: Path,
    aggregated_at: datetime,
    round_base_snapshot_cache: RoundBaseSnapshotCache | None = None,
) -> LoraClassifierMaterializedState:
    """server-published PEFT-backed classifier base state를 materialize한다."""

    if round_base_snapshot_cache is None:
        return _materialize_peft_encoder_base_parameters(
            active_adapter_state=active_adapter_state,
            output_dir=output_dir,
            aggregated_at=aggregated_at,
        )
    snapshot = round_base_snapshot_cache.get_or_materialize(
        key=_peft_encoder_cache_key(active_adapter_state),
        materialize=lambda: _materialize_peft_encoder_base_parameters(
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


def load_peft_encoder_base_partition_parameters(
    *,
    active_adapter_state: PeftEncoderState,
    output_dir: Path,
    aggregated_at: datetime,
    round_base_snapshot_cache: RoundBaseSnapshotCache | None = None,
) -> dict[str, LoraClassifierMaterializedState]:
    """server-published PEFT encoder partitioned base state를 materialize한다."""

    if round_base_snapshot_cache is None:
        return _materialize_peft_encoder_base_partition_parameters(
            active_adapter_state=active_adapter_state,
            output_dir=output_dir,
            aggregated_at=aggregated_at,
        )
    snapshot = round_base_snapshot_cache.get_or_materialize(
        key=_peft_encoder_partition_cache_key(active_adapter_state),
        materialize=lambda: _materialize_peft_encoder_base_partition_parameters(
            active_adapter_state=active_adapter_state,
            output_dir=output_dir,
            aggregated_at=aggregated_at,
        ),
    )
    if not isinstance(snapshot, dict):
        raise TypeError(
            "Round base snapshot cache returned unexpected LoRA-classifier "
            f"partition state: {type(snapshot)!r}."
        )
    return snapshot


def _materialize_peft_encoder_base_parameters(
    *,
    active_adapter_state: PeftEncoderState,
    output_dir: Path,
    aggregated_at: datetime,
) -> LoraClassifierMaterializedState:
    return materialize_base_peft_encoder_state(
        base_state=active_adapter_state,
        context=FederatedAggregationContext(
            next_model_revision=active_adapter_state.model_revision,
            aggregated_at=aggregated_at,
            artifact_loader=AggregationArtifactStore(
                state_root=output_dir / "main_server" / "aggregation_artifacts"
            ),
        ),
    )


def _materialize_peft_encoder_base_partition_parameters(
    *,
    active_adapter_state: PeftEncoderState,
    output_dir: Path,
    aggregated_at: datetime,
) -> dict[str, LoraClassifierMaterializedState]:
    return materialize_base_peft_encoder_partitioned_state(
        base_state=active_adapter_state,
        context=FederatedAggregationContext(
            next_model_revision=active_adapter_state.model_revision,
            aggregated_at=aggregated_at,
            artifact_loader=AggregationArtifactStore(
                state_root=output_dir / "main_server" / "aggregation_artifacts"
            ),
        ),
    )


def _peft_encoder_cache_key(
    active_adapter_state: PeftEncoderState,
) -> RoundBaseSnapshotCacheKey:
    return RoundBaseSnapshotCacheKey(
        adapter_kind=str(active_adapter_state.adapter_kind),
        model_revision=str(active_adapter_state.model_revision),
        schema_version=str(active_adapter_state.schema_version),
        artifact_refs=peft_encoder_base_state_artifact_refs(active_adapter_state),
        materializer_name=PEFT_ENCODER_BASE_STATE_MATERIALIZER_NAME,
    )


def _peft_encoder_partition_cache_key(
    active_adapter_state: PeftEncoderState,
) -> RoundBaseSnapshotCacheKey:
    return RoundBaseSnapshotCacheKey(
        adapter_kind=str(active_adapter_state.adapter_kind),
        model_revision=str(active_adapter_state.model_revision),
        schema_version=str(active_adapter_state.schema_version),
        artifact_refs=peft_encoder_base_state_artifact_refs(active_adapter_state),
        materializer_name=PEFT_ENCODER_BASE_PARTITIONED_STATE_MATERIALIZER_NAME,
    )


load_lora_classifier_base_parameters = load_peft_encoder_base_parameters
load_lora_classifier_base_partition_parameters = (
    load_peft_encoder_base_partition_parameters
)
_materialize_lora_classifier_base_parameters = _materialize_peft_encoder_base_parameters
_materialize_lora_classifier_base_partition_parameters = (
    _materialize_peft_encoder_base_partition_parameters
)
