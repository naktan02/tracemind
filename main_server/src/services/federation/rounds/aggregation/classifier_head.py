"""Classifier-head aggregation backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from main_server.src.services.federation.rounds.aggregation.models import (
    AggregationResult,
)
from methods.federated.aggregation.fedavg.classifier_head_fedavg import (
    ClassifierHeadFedAvgUpdate,
    compute_classifier_head_fedavg,
)
from shared.src.contracts.adapter_contracts import (
    ClassifierHeadDelta,
    ClassifierHeadState,
)
from shared.src.contracts.adapter_family_metadata import CLASSIFIER_HEAD_FAMILY_METADATA
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from .diagonal_scale_defaults import AggregationConfigScalar
from .registry import register_shared_adapter_aggregation_backend
from .runtime_adapter import (
    require_base_adapter_kind,
    require_update_matches_base,
    select_non_empty_updates,
)


@dataclass(slots=True)
class ClassifierHeadFedAvgAggregationService:
    """Classifier-head server boundary를 FedAvg method core에 연결한다."""

    adapter_kind: str = CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        next_model_revision: str,
        aggregated_at: datetime,
    ) -> AggregationResult:
        if not isinstance(base_state, ClassifierHeadState):
            raise TypeError(
                "ClassifierHeadFedAvgAggregationService expects "
                f"ClassifierHeadState as the base state, got {type(base_state)!r}."
            )
        require_base_adapter_kind(
            base_state=base_state,
            adapter_kind=self.adapter_kind,
            context="classifier-head",
        )

        valid_updates = select_non_empty_updates(update_payloads)

        labels = base_state.labels
        embedding_dim = base_state.embedding_dim
        method_updates: list[ClassifierHeadFedAvgUpdate] = []

        for payload in valid_updates:
            if not isinstance(payload, ClassifierHeadDelta):
                raise TypeError(
                    "ClassifierHeadFedAvgAggregationService expects "
                    f"ClassifierHeadDelta updates, got {type(payload)!r}."
                )
            require_update_matches_base(
                payload=payload,
                base_state=base_state,
                adapter_kind=self.adapter_kind,
                context="classifier-head",
            )
            if payload.labels != labels:
                raise ValueError(
                    "Classifier head updates must share the same ordered labels."
                )
            if payload.embedding_dim != embedding_dim:
                raise ValueError(
                    "All update payloads must share the same embedding_dim."
                )
            method_updates.append(
                ClassifierHeadFedAvgUpdate(
                    label_weight_deltas=payload.label_weight_deltas,
                    label_bias_deltas=payload.label_bias_deltas,
                    example_count=payload.example_count,
                    mean_confidence=payload.mean_confidence,
                    mean_margin=payload.mean_margin,
                    delta_l2_norm=payload.l2_norm(),
                )
            )

        method_result = compute_classifier_head_fedavg(
            base_label_weights=base_state.label_weights,
            base_label_biases=base_state.label_biases,
            updates=method_updates,
        )

        next_state = ClassifierHeadState(
            schema_version=base_state.schema_version,
            adapter_kind=base_state.adapter_kind,
            model_id=base_state.model_id,
            model_revision=next_model_revision,
            training_scope=base_state.training_scope,
            updated_at=aggregated_at,
            label_weights=method_result.label_weights,
            label_biases=method_result.label_biases,
        )
        return AggregationResult(
            next_state=next_state,
            aggregated_metrics=method_result.aggregated_metrics,
            update_count=method_result.update_count,
        )


@register_shared_adapter_aggregation_backend(
    CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,
    "fedavg",
    "classifier_head_fedavg",
    catalog_entry=RegistryCatalogEntry(
        item_name=f"{CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind}.fedavg",
        display_name="fedavg",
        implementation_module=compute_classifier_head_fedavg.__module__,
        core_method_name="fedavg",
        family_name=CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,
        supported_adapter_kinds=(CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,),
        metadata={"adapter_kind": CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind},
    ),
)
def build_classifier_head_fedavg_aggregation_backend(
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> ClassifierHeadFedAvgAggregationService:
    """registry용 classifier-head FedAvg backend factory."""

    del overrides
    return ClassifierHeadFedAvgAggregationService()
