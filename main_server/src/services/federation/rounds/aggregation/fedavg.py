"""FedAvg server aggregation runtime adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from methods.federated.aggregation.fedavg.classifier_head_fedavg import (
    ClassifierHeadFedAvgUpdate,
    compute_classifier_head_fedavg,
)
from methods.federated.aggregation.fedavg.diagonal_scale_fedavg import (
    DiagonalScaleFedAvgUpdate,
    compute_diagonal_scale_fedavg,
)
from methods.federated.aggregation.fedavg.lora_classifier_fedavg import (
    LoraClassifierFedAvgUpdate,
    compute_lora_classifier_fedavg,
)
from shared.src.contracts.adapter_contracts import (
    ClassifierHeadDelta,
    ClassifierHeadState,
    LoraClassifierDelta,
    LoraClassifierState,
    VectorAdapterDelta,
    VectorAdapterState,
)
from shared.src.contracts.adapter_family_metadata import (
    CLASSIFIER_HEAD_FAMILY_METADATA,
    DIAGONAL_SCALE_FAMILY_METADATA,
    LORA_CLASSIFIER_FAMILY_METADATA,
)
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from .artifact_refs import AggregatedArtifactRefBuilder
from .models import AggregationConfigScalar, AggregationResult
from .registry import register_shared_adapter_aggregation_backend
from .runtime_adapter import (
    require_typed_base_state,
    select_validated_updates,
)

DEFAULT_DIAGONAL_SCALE_MIN_SCALE = 0.75
DEFAULT_DIAGONAL_SCALE_MAX_SCALE = 1.25
DEFAULT_AGGREGATED_LORA_ARTIFACT_REF_PREFIX = "server-aggregate://lora_classifier"
DEFAULT_AGGREGATED_LORA_ARTIFACT_FORMAT = "server_aggregated_artifact_ref"

FedAvgAggregate = Callable[
    [
        SharedAdapterState,
        Sequence[SharedAdapterUpdate],
        str,
        datetime,
        Mapping[str, AggregationConfigScalar] | None,
    ],
    AggregationResult,
]


@dataclass(frozen=True, slots=True)
class FedAvgAdapterRuntimeSpec:
    """FedAvg가 한 adapter family payload를 처리하는 runtime-boundary spec."""

    adapter_kind: str
    state_type: type[object]
    update_type: type[object]
    context: str
    backend_names: tuple[str, ...]
    implementation_module: str
    metadata: Mapping[str, AggregationConfigScalar | None]
    aggregate: FedAvgAggregate


@dataclass(slots=True)
class FedAvgAggregationRuntime:
    """Adapter family contract를 FedAvg method core에 연결하는 generic runtime."""

    spec: FedAvgAdapterRuntimeSpec
    overrides: Mapping[str, AggregationConfigScalar] | None = None

    @property
    def adapter_kind(self) -> str:
        """이 runtime이 처리하는 shared adapter family discriminator."""

        return self.spec.adapter_kind

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        next_model_revision: str,
        aggregated_at: datetime,
    ) -> AggregationResult:
        """공통 lineage/type 검증 후 family별 FedAvg payload adapter를 호출한다."""

        typed_base_state = require_typed_base_state(
            base_state=base_state,
            state_type=self.spec.state_type,
            adapter_kind=self.spec.adapter_kind,
            context=self.spec.context,
            service_name="FedAvgAggregationRuntime",
        )
        typed_updates = select_validated_updates(
            update_payloads,
            update_type=self.spec.update_type,
            base_state=typed_base_state,
            adapter_kind=self.spec.adapter_kind,
            context=self.spec.context,
            service_name="FedAvgAggregationRuntime",
        )
        return self.spec.aggregate(
            typed_base_state,
            typed_updates,
            next_model_revision,
            aggregated_at,
            self.overrides,
        )


def _read_float(
    source: Mapping[str, AggregationConfigScalar] | None,
    key: str,
    default: float,
) -> float:
    if source is None:
        return default
    value = source.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must not be bool.")
    return float(value)


def _aggregate_diagonal_scale_fedavg(
    base_state: SharedAdapterState,
    update_payloads: Sequence[SharedAdapterUpdate],
    next_model_revision: str,
    aggregated_at: datetime,
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> AggregationResult:
    base_state = cast(VectorAdapterState, base_state)
    updates = [cast(VectorAdapterDelta, payload) for payload in update_payloads]
    embedding_dim = base_state.embedding_dim
    method_updates: list[DiagonalScaleFedAvgUpdate] = []

    for payload in updates:
        if payload.embedding_dim != embedding_dim:
            raise ValueError("All update payloads must share the same embedding_dim.")
        method_updates.append(
            DiagonalScaleFedAvgUpdate(
                dimension_deltas=payload.dimension_deltas,
                example_count=payload.example_count,
                mean_confidence=payload.mean_confidence,
                mean_margin=payload.mean_margin,
                delta_l2_norm=payload.l2_norm(),
            )
        )

    method_result = compute_diagonal_scale_fedavg(
        base_dimension_scales=base_state.dimension_scales,
        updates=method_updates,
        min_scale=_read_float(
            overrides,
            "min_scale",
            DEFAULT_DIAGONAL_SCALE_MIN_SCALE,
        ),
        max_scale=_read_float(
            overrides,
            "max_scale",
            DEFAULT_DIAGONAL_SCALE_MAX_SCALE,
        ),
    )
    next_state = VectorAdapterState(
        schema_version=base_state.schema_version,
        model_id=base_state.model_id,
        model_revision=next_model_revision,
        training_scope=base_state.training_scope,
        dimension_scales=method_result.next_dimension_scales,
        updated_at=aggregated_at,
        adapter_kind=base_state.adapter_kind,
    )
    return AggregationResult(
        next_state=next_state,
        aggregated_metrics=method_result.aggregated_metrics,
        update_count=method_result.update_count,
    )


def _aggregate_classifier_head_fedavg(
    base_state: SharedAdapterState,
    update_payloads: Sequence[SharedAdapterUpdate],
    next_model_revision: str,
    aggregated_at: datetime,
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> AggregationResult:
    del overrides

    base_state = cast(ClassifierHeadState, base_state)
    updates = [cast(ClassifierHeadDelta, payload) for payload in update_payloads]
    labels = base_state.labels
    embedding_dim = base_state.embedding_dim
    method_updates: list[ClassifierHeadFedAvgUpdate] = []

    for payload in updates:
        if payload.labels != labels:
            raise ValueError(
                "Classifier head updates must share the same ordered labels."
            )
        if payload.embedding_dim != embedding_dim:
            raise ValueError("All update payloads must share the same embedding_dim.")
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


def _aggregate_lora_classifier_fedavg(
    base_state: SharedAdapterState,
    update_payloads: Sequence[SharedAdapterUpdate],
    next_model_revision: str,
    aggregated_at: datetime,
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> AggregationResult:
    base_state = cast(LoraClassifierState, base_state)
    updates = [cast(LoraClassifierDelta, payload) for payload in update_payloads]
    method_updates = [
        _to_lora_classifier_method_update(base_state=base_state, payload=payload)
        for payload in updates
    ]
    method_result = compute_lora_classifier_fedavg(
        label_schema=base_state.label_schema,
        updates=method_updates,
    )
    artifact_refs = _build_aggregated_lora_artifact_ref_builder(
        overrides
    ).build_lora_classifier_refs(next_model_revision=next_model_revision)
    next_state = LoraClassifierState(
        schema_version=base_state.schema_version,
        adapter_kind=base_state.adapter_kind,
        model_id=base_state.model_id,
        model_revision=next_model_revision,
        training_scope=base_state.training_scope,
        updated_at=aggregated_at,
        backbone=base_state.backbone,
        lora_config=base_state.lora_config,
        label_schema=base_state.label_schema,
        lora_adapter_artifact_ref=artifact_refs.lora_adapter_artifact_ref,
        classifier_head_artifact_ref=artifact_refs.classifier_head_artifact_ref,
        artifact_format=artifact_refs.artifact_format,
    )
    return AggregationResult(
        next_state=next_state,
        aggregated_metrics=method_result.aggregated_metrics,
        update_count=method_result.update_count,
    )


def _to_lora_classifier_method_update(
    *,
    base_state: LoraClassifierState,
    payload: LoraClassifierDelta,
) -> LoraClassifierFedAvgUpdate:
    if _payload_snapshot(payload.backbone) != _payload_snapshot(base_state.backbone):
        raise ValueError("All LoRA-classifier updates must match the backbone.")
    if _payload_snapshot(payload.lora_config) != _payload_snapshot(
        base_state.lora_config
    ):
        raise ValueError("All LoRA-classifier updates must match the LoRA config.")
    if payload.labels != base_state.labels:
        raise ValueError(
            "LoRA-classifier updates must share the base ordered label_schema."
        )
    if (
        payload.lora_parameter_deltas is None
        or payload.classifier_head_weight_deltas is None
    ):
        raise ValueError(
            "LoRA-classifier FedAvg currently requires inline "
            "lora_parameter_deltas and classifier_head_weight_deltas. "
            "Artifact-ref-only updates require a server artifact materializer."
        )
    return LoraClassifierFedAvgUpdate(
        lora_parameter_deltas=payload.lora_parameter_deltas,
        classifier_head_weight_deltas=payload.classifier_head_weight_deltas,
        classifier_head_bias_deltas=payload.classifier_head_bias_deltas,
        example_count=payload.example_count,
        mean_confidence=payload.mean_confidence,
        mean_margin=payload.mean_margin,
        delta_l2_norm=payload.l2_norm(),
    )


def _build_aggregated_lora_artifact_ref_builder(
    source: Mapping[str, AggregationConfigScalar] | None,
) -> AggregatedArtifactRefBuilder:
    if source is not None:
        unknown_keys = sorted(set(source) - {"artifact_ref_prefix", "artifact_format"})
        if unknown_keys:
            raise ValueError(
                "Unsupported LoRA-classifier aggregate artifact config key(s): "
                f"{unknown_keys}."
            )
    return AggregatedArtifactRefBuilder(
        artifact_ref_prefix=str(
            (source or {}).get(
                "artifact_ref_prefix",
                DEFAULT_AGGREGATED_LORA_ARTIFACT_REF_PREFIX,
            )
        ).strip(),
        artifact_format=str(
            (source or {}).get(
                "artifact_format",
                DEFAULT_AGGREGATED_LORA_ARTIFACT_FORMAT,
            )
        ).strip(),
    )


def _payload_snapshot(payload) -> dict[str, object]:
    return payload.model_dump(mode="json")


DIAGONAL_SCALE_FEDAVG_SPEC = FedAvgAdapterRuntimeSpec(
    adapter_kind=DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,
    state_type=VectorAdapterState,
    update_type=VectorAdapterDelta,
    context="diagonal scale",
    backend_names=("fedavg", "diagonal_scale_fedavg"),
    implementation_module=compute_diagonal_scale_fedavg.__module__,
    metadata={"adapter_kind": DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind},
    aggregate=_aggregate_diagonal_scale_fedavg,
)

CLASSIFIER_HEAD_FEDAVG_SPEC = FedAvgAdapterRuntimeSpec(
    adapter_kind=CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,
    state_type=ClassifierHeadState,
    update_type=ClassifierHeadDelta,
    context="classifier-head",
    backend_names=("fedavg", "classifier_head_fedavg"),
    implementation_module=compute_classifier_head_fedavg.__module__,
    metadata={"adapter_kind": CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind},
    aggregate=_aggregate_classifier_head_fedavg,
)

LORA_CLASSIFIER_FEDAVG_SPEC = FedAvgAdapterRuntimeSpec(
    adapter_kind=LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind,
    state_type=LoraClassifierState,
    update_type=LoraClassifierDelta,
    context="LoRA-classifier",
    backend_names=("fedavg", "lora_classifier_fedavg"),
    implementation_module=compute_lora_classifier_fedavg.__module__,
    metadata={
        "adapter_kind": LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind,
        "requires_inline_or_materialized_artifacts": True,
    },
    aggregate=_aggregate_lora_classifier_fedavg,
)


def _build_fedavg_runtime_factory(
    spec: FedAvgAdapterRuntimeSpec,
) -> Callable[[Mapping[str, AggregationConfigScalar] | None], FedAvgAggregationRuntime]:
    def _factory(
        overrides: Mapping[str, AggregationConfigScalar] | None,
    ) -> FedAvgAggregationRuntime:
        return FedAvgAggregationRuntime(spec=spec, overrides=overrides)

    return _factory


def _register_fedavg_runtime(spec: FedAvgAdapterRuntimeSpec) -> None:
    register_shared_adapter_aggregation_backend(
        spec.adapter_kind,
        *spec.backend_names,
        catalog_entry=RegistryCatalogEntry(
            item_name=f"{spec.adapter_kind}.fedavg",
            display_name="fedavg",
            implementation_module=spec.implementation_module,
            core_method_name="fedavg",
            family_name=spec.adapter_kind,
            supported_adapter_kinds=(spec.adapter_kind,),
            metadata=spec.metadata,
        ),
        factory=_build_fedavg_runtime_factory(spec),
    )


for _SPEC in (
    DIAGONAL_SCALE_FEDAVG_SPEC,
    CLASSIFIER_HEAD_FEDAVG_SPEC,
    LORA_CLASSIFIER_FEDAVG_SPEC,
):
    _register_fedavg_runtime(_SPEC)
