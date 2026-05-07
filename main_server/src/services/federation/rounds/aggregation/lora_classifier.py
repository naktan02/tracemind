"""LoRA-classifier aggregation backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from methods.federated.aggregation.fedavg.lora_classifier_fedavg import (
    LoraClassifierFedAvgUpdate,
    compute_lora_classifier_fedavg,
)
from shared.src.contracts.adapter_contracts import (
    LoraClassifierDelta,
    LoraClassifierState,
)
from shared.src.contracts.adapter_family_metadata import LORA_CLASSIFIER_FAMILY_METADATA
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from .diagonal_scale_defaults import AggregationConfigScalar
from .models import AggregationResult
from .registry import register_shared_adapter_aggregation_backend
from .runtime_adapter import (
    require_base_adapter_kind,
    require_update_matches_base,
    select_non_empty_updates,
)


@dataclass(frozen=True, slots=True)
class LoraClassifierFedAvgAggregationConfig:
    """LoRA-classifier FedAvg 서버 집계 설정."""

    artifact_ref_prefix: str = "server-aggregate://lora_classifier"
    artifact_format: str = "server_aggregated_artifact_ref"

    def __post_init__(self) -> None:
        _require_non_empty_str(
            self.artifact_ref_prefix,
            field_name="artifact_ref_prefix",
        )
        _require_non_empty_str(self.artifact_format, field_name="artifact_format")

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, AggregationConfigScalar] | None,
    ) -> "LoraClassifierFedAvgAggregationConfig":
        if source is None:
            return cls()
        unknown_keys = sorted(set(source) - {"artifact_ref_prefix", "artifact_format"})
        if unknown_keys:
            raise ValueError(
                "Unsupported LoRA-classifier FedAvg aggregation config key(s): "
                f"{unknown_keys}."
            )
        defaults = cls()
        return cls(
            artifact_ref_prefix=str(
                source.get("artifact_ref_prefix", defaults.artifact_ref_prefix)
            ).strip(),
            artifact_format=str(
                source.get("artifact_format", defaults.artifact_format)
            ).strip(),
        )


@dataclass(slots=True)
class LoraClassifierFedAvgAggregationService:
    """LoRA-classifier server boundary를 FedAvg method core에 연결한다."""

    adapter_kind: str = LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind
    config: LoraClassifierFedAvgAggregationConfig = field(
        default_factory=LoraClassifierFedAvgAggregationConfig
    )

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, AggregationConfigScalar] | None,
    ) -> "LoraClassifierFedAvgAggregationService":
        return cls(config=LoraClassifierFedAvgAggregationConfig.from_mapping(source))

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        next_model_revision: str,
        aggregated_at: datetime,
    ) -> AggregationResult:
        if not isinstance(base_state, LoraClassifierState):
            raise TypeError(
                "LoraClassifierFedAvgAggregationService expects "
                f"LoraClassifierState as the base state, got {type(base_state)!r}."
            )
        require_base_adapter_kind(
            base_state=base_state,
            adapter_kind=self.adapter_kind,
            context="LoRA-classifier",
        )

        valid_updates = select_non_empty_updates(update_payloads)

        method_updates = [
            self._to_method_update(base_state=base_state, payload=payload)
            for payload in valid_updates
        ]
        method_result = compute_lora_classifier_fedavg(
            label_schema=base_state.label_schema,
            updates=method_updates,
        )
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
            lora_adapter_artifact_ref=_build_aggregate_artifact_ref(
                prefix=self.config.artifact_ref_prefix,
                next_model_revision=next_model_revision,
                artifact_name="lora_adapter",
            ),
            classifier_head_artifact_ref=_build_aggregate_artifact_ref(
                prefix=self.config.artifact_ref_prefix,
                next_model_revision=next_model_revision,
                artifact_name="classifier_head",
            ),
            artifact_format=self.config.artifact_format,
        )
        return AggregationResult(
            next_state=next_state,
            aggregated_metrics=method_result.aggregated_metrics,
            update_count=method_result.update_count,
        )

    def _to_method_update(
        self,
        *,
        base_state: LoraClassifierState,
        payload: SharedAdapterUpdate,
    ) -> LoraClassifierFedAvgUpdate:
        if not isinstance(payload, LoraClassifierDelta):
            raise TypeError(
                "LoraClassifierFedAvgAggregationService expects "
                f"LoraClassifierDelta updates, got {type(payload)!r}."
            )
        require_update_matches_base(
            payload=payload,
            base_state=base_state,
            adapter_kind=self.adapter_kind,
            context="LoRA-classifier",
        )
        if _payload_snapshot(payload.backbone) != _payload_snapshot(
            base_state.backbone
        ):
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


def _payload_snapshot(payload) -> dict[str, object]:
    return payload.model_dump(mode="json")


def _build_aggregate_artifact_ref(
    *,
    prefix: str,
    next_model_revision: str,
    artifact_name: str,
) -> str:
    return "/".join(
        (
            prefix.rstrip("/"),
            _slug_ref_part(next_model_revision),
            _slug_ref_part(artifact_name),
        )
    )


def _slug_ref_part(value: str) -> str:
    normalized = value.strip().replace("/", "_")
    if not normalized:
        raise ValueError("artifact ref path parts must not be empty.")
    return normalized


def _require_non_empty_str(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty.")


@register_shared_adapter_aggregation_backend(
    LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind,
    "fedavg",
    "lora_classifier_fedavg",
    catalog_entry=RegistryCatalogEntry(
        item_name=f"{LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind}.fedavg",
        display_name="fedavg",
        implementation_module=compute_lora_classifier_fedavg.__module__,
        core_method_name="fedavg",
        family_name=LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind,
        supported_adapter_kinds=(LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind,),
        metadata={
            "adapter_kind": LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind,
            "requires_inline_or_materialized_artifacts": True,
        },
    ),
)
def build_lora_classifier_fedavg_aggregation_backend(
    overrides: Mapping[str, AggregationConfigScalar] | None,
) -> LoraClassifierFedAvgAggregationService:
    """registry용 LoRA-classifier FedAvg backend factory."""

    return LoraClassifierFedAvgAggregationService.from_mapping(overrides)
