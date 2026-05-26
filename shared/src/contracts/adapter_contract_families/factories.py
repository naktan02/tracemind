"""Shared adapter payload factories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from shared.src.contracts.common_types import TrainingScope
from shared.src.contracts.model_contracts import ModelManifestPayload

from .base import (
    CLASSIFIER_HEAD_DELTA_V1,
    CURRENT_SHARED_ADAPTER_STATE_V1,
    LORA_CLASSIFIER_DELTA_V1,
    LORA_CLASSIFIER_STATE_V1,
    PEFT_CLASSIFIER_DELTA_V2,
    PEFT_CLASSIFIER_STATE_V2,
    VECTOR_ADAPTER_DELTA_V1,
    AdapterKind,
    CurrentSharedAdapterStatePayload,
    SharedAdapterStatePayload,
)
from .classifier_head import (
    ClassifierHeadAdapterStatePayload,
    ClassifierHeadAdapterUpdatePayload,
)
from .diagonal_scale import (
    DiagonalScaleAdapterStatePayload,
    DiagonalScaleAdapterUpdatePayload,
)
from .lora_classifier import (
    LoraClassifierAdapterStatePayload,
    LoraClassifierAdapterUpdatePayload,
    LoraClassifierBackbonePayload,
    LoraClassifierConfigPayload,
    LoraClassifierPartitionDeltaPayload,
)
from .peft_classifier import (
    PeftAdapterConfigPayload,
    PeftClassifierAdapterStatePayload,
    PeftClassifierAdapterUpdatePayload,
    PeftClassifierBackbonePayload,
    PeftClassifierPartitionDeltaPayload,
)


def make_identity_state_payload(
    *,
    model_id: str,
    model_revision: str,
    embedding_dim: int,
    training_scope: TrainingScope = TrainingScope.ADAPTER_ONLY,
    updated_at: datetime | None = None,
) -> DiagonalScaleAdapterStatePayload:
    """모든 차원 scale=1.0 인 identity adapter state payload를 만든다."""
    return DiagonalScaleAdapterStatePayload.identity(
        model_id=model_id,
        model_revision=model_revision,
        training_scope=TrainingScope(training_scope),
        embedding_dim=embedding_dim,
        updated_at=updated_at or datetime.now(tz=timezone.utc),
    )


def make_diagonal_delta_payload(
    *,
    model_id: str,
    base_model_revision: str,
    dimension_deltas: list[float],
    example_count: int,
    mean_confidence: float,
    training_scope: TrainingScope = TrainingScope.ADAPTER_ONLY,
    mean_margin: float | None = None,
    label_counts: dict[str, int] | None = None,
    created_at: datetime | None = None,
) -> DiagonalScaleAdapterUpdatePayload:
    """diagonal scale adapter update payload를 만드는 표준 factory."""
    return DiagonalScaleAdapterUpdatePayload(
        schema_version=VECTOR_ADAPTER_DELTA_V1,
        adapter_kind=AdapterKind.DIAGONAL_SCALE.value,
        model_id=model_id,
        base_model_revision=base_model_revision,
        training_scope=training_scope,
        example_count=example_count,
        created_at=created_at or datetime.now(tz=timezone.utc),
        dimension_deltas=dimension_deltas,
        mean_confidence=mean_confidence,
        mean_margin=mean_margin,
        label_counts=label_counts or {},
    )


def make_zero_classifier_head_state_payload(
    *,
    model_id: str,
    model_revision: str,
    labels: Sequence[str],
    embedding_dim: int,
    training_scope: TrainingScope = TrainingScope.HEAD_ONLY,
    updated_at: datetime | None = None,
) -> ClassifierHeadAdapterStatePayload:
    """0으로 초기화된 classifier head state payload를 만든다."""
    return ClassifierHeadAdapterStatePayload.zero_initialized(
        model_id=model_id,
        model_revision=model_revision,
        labels=labels,
        embedding_dim=embedding_dim,
        training_scope=TrainingScope(training_scope),
        updated_at=updated_at or datetime.now(tz=timezone.utc),
    )


def make_classifier_head_delta_payload(
    *,
    model_id: str,
    base_model_revision: str,
    label_weight_deltas: dict[str, list[float]],
    example_count: int,
    mean_confidence: float,
    training_scope: TrainingScope = TrainingScope.HEAD_ONLY,
    label_bias_deltas: dict[str, float] | None = None,
    mean_margin: float | None = None,
    label_counts: dict[str, int] | None = None,
    created_at: datetime | None = None,
) -> ClassifierHeadAdapterUpdatePayload:
    """classifier head update payload를 만드는 표준 factory."""
    return ClassifierHeadAdapterUpdatePayload(
        schema_version=CLASSIFIER_HEAD_DELTA_V1,
        adapter_kind=AdapterKind.CLASSIFIER_HEAD.value,
        model_id=model_id,
        base_model_revision=base_model_revision,
        training_scope=training_scope,
        example_count=example_count,
        created_at=created_at or datetime.now(tz=timezone.utc),
        label_weight_deltas=label_weight_deltas,
        label_bias_deltas=label_bias_deltas or {},
        mean_confidence=mean_confidence,
        mean_margin=mean_margin,
        label_counts=label_counts or {},
    )


def make_lora_classifier_state_payload(
    *,
    model_id: str,
    model_revision: str,
    backbone: LoraClassifierBackbonePayload | Mapping[str, object],
    lora_config: LoraClassifierConfigPayload | Mapping[str, object],
    label_schema: Sequence[str],
    training_scope: TrainingScope = TrainingScope.ADAPTER_ONLY,
    lora_adapter_artifact_ref: str | None = None,
    classifier_head_artifact_ref: str | None = None,
    artifact_format: str = "artifact_ref",
    updated_at: datetime | None = None,
) -> LoraClassifierAdapterStatePayload:
    """LoRA-classifier state payload를 만드는 표준 factory."""
    return LoraClassifierAdapterStatePayload(
        schema_version=LORA_CLASSIFIER_STATE_V1,
        adapter_kind=AdapterKind.LORA_CLASSIFIER.value,
        model_id=model_id,
        model_revision=model_revision,
        training_scope=training_scope,
        updated_at=updated_at or datetime.now(tz=timezone.utc),
        backbone=backbone,
        lora_config=lora_config,
        label_schema=list(label_schema),
        lora_adapter_artifact_ref=lora_adapter_artifact_ref,
        classifier_head_artifact_ref=classifier_head_artifact_ref,
        artifact_format=artifact_format,
    )


def make_lora_classifier_delta_payload(
    *,
    model_id: str,
    base_model_revision: str,
    backbone: LoraClassifierBackbonePayload | Mapping[str, object],
    lora_config: LoraClassifierConfigPayload | Mapping[str, object],
    label_schema: Sequence[str],
    example_count: int,
    training_scope: TrainingScope = TrainingScope.ADAPTER_ONLY,
    lora_delta_artifact_ref: str | None = None,
    classifier_head_delta_artifact_ref: str | None = None,
    lora_parameter_deltas: dict[str, list[float]] | None = None,
    classifier_head_weight_deltas: dict[str, list[float]] | None = None,
    classifier_head_bias_deltas: dict[str, float] | None = None,
    partitioned_deltas: (
        dict[str, LoraClassifierPartitionDeltaPayload | Mapping[str, object]] | None
    ) = None,
    partitioned_deltas_artifact_ref: str | None = None,
    delta_format: str = "artifact_ref",
    mean_confidence: float | None = None,
    mean_margin: float | None = None,
    label_counts: dict[str, int] | None = None,
    delta_l2_norm: float | None = None,
    created_at: datetime | None = None,
) -> LoraClassifierAdapterUpdatePayload:
    """LoRA-classifier update payload를 만드는 표준 factory."""
    return LoraClassifierAdapterUpdatePayload(
        schema_version=LORA_CLASSIFIER_DELTA_V1,
        adapter_kind=AdapterKind.LORA_CLASSIFIER.value,
        model_id=model_id,
        base_model_revision=base_model_revision,
        training_scope=training_scope,
        example_count=example_count,
        created_at=created_at or datetime.now(tz=timezone.utc),
        backbone=backbone,
        lora_config=lora_config,
        label_schema=list(label_schema),
        lora_delta_artifact_ref=lora_delta_artifact_ref,
        classifier_head_delta_artifact_ref=classifier_head_delta_artifact_ref,
        lora_parameter_deltas=lora_parameter_deltas,
        classifier_head_weight_deltas=classifier_head_weight_deltas,
        classifier_head_bias_deltas=classifier_head_bias_deltas or {},
        partitioned_deltas=partitioned_deltas,
        partitioned_deltas_artifact_ref=partitioned_deltas_artifact_ref,
        delta_format=delta_format,
        mean_confidence=mean_confidence,
        mean_margin=mean_margin,
        label_counts=label_counts or {},
        delta_l2_norm=delta_l2_norm,
    )


def make_peft_classifier_state_payload(
    *,
    model_id: str,
    model_revision: str,
    backbone: PeftClassifierBackbonePayload | Mapping[str, object],
    peft_adapter_config: PeftAdapterConfigPayload | Mapping[str, object],
    label_schema: Sequence[str],
    training_scope: TrainingScope = TrainingScope.ADAPTER_ONLY,
    peft_adapter_artifact_ref: str | None = None,
    classifier_head_artifact_ref: str | None = None,
    artifact_format: str = "artifact_ref",
    updated_at: datetime | None = None,
) -> PeftClassifierAdapterStatePayload:
    """PEFT-classifier state payload를 만드는 표준 factory."""
    return PeftClassifierAdapterStatePayload(
        schema_version=PEFT_CLASSIFIER_STATE_V2,
        adapter_kind=AdapterKind.PEFT_CLASSIFIER.value,
        model_id=model_id,
        model_revision=model_revision,
        training_scope=training_scope,
        updated_at=updated_at or datetime.now(tz=timezone.utc),
        backbone=backbone,
        peft_adapter_config=peft_adapter_config,
        label_schema=list(label_schema),
        peft_adapter_artifact_ref=peft_adapter_artifact_ref,
        classifier_head_artifact_ref=classifier_head_artifact_ref,
        artifact_format=artifact_format,
    )


def make_peft_classifier_delta_payload(
    *,
    model_id: str,
    base_model_revision: str,
    backbone: PeftClassifierBackbonePayload | Mapping[str, object],
    peft_adapter_config: PeftAdapterConfigPayload | Mapping[str, object],
    label_schema: Sequence[str],
    example_count: int,
    training_scope: TrainingScope = TrainingScope.ADAPTER_ONLY,
    peft_adapter_delta_artifact_ref: str | None = None,
    classifier_head_delta_artifact_ref: str | None = None,
    peft_parameter_deltas: dict[str, list[float]] | None = None,
    classifier_head_weight_deltas: dict[str, list[float]] | None = None,
    classifier_head_bias_deltas: dict[str, float] | None = None,
    partitioned_deltas: (
        dict[str, PeftClassifierPartitionDeltaPayload | Mapping[str, object]] | None
    ) = None,
    partitioned_deltas_artifact_ref: str | None = None,
    delta_format: str = "artifact_ref",
    mean_confidence: float | None = None,
    mean_margin: float | None = None,
    label_counts: dict[str, int] | None = None,
    delta_l2_norm: float | None = None,
    created_at: datetime | None = None,
) -> PeftClassifierAdapterUpdatePayload:
    """PEFT-classifier update payload를 만드는 표준 factory."""
    return PeftClassifierAdapterUpdatePayload(
        schema_version=PEFT_CLASSIFIER_DELTA_V2,
        adapter_kind=AdapterKind.PEFT_CLASSIFIER.value,
        model_id=model_id,
        base_model_revision=base_model_revision,
        training_scope=training_scope,
        example_count=example_count,
        created_at=created_at or datetime.now(tz=timezone.utc),
        backbone=backbone,
        peft_adapter_config=peft_adapter_config,
        label_schema=list(label_schema),
        peft_adapter_delta_artifact_ref=peft_adapter_delta_artifact_ref,
        classifier_head_delta_artifact_ref=classifier_head_delta_artifact_ref,
        peft_parameter_deltas=peft_parameter_deltas,
        classifier_head_weight_deltas=classifier_head_weight_deltas,
        classifier_head_bias_deltas=classifier_head_bias_deltas or {},
        partitioned_deltas=partitioned_deltas,
        partitioned_deltas_artifact_ref=partitioned_deltas_artifact_ref,
        delta_format=delta_format,
        mean_confidence=mean_confidence,
        mean_margin=mean_margin,
        label_counts=label_counts or {},
        delta_l2_norm=delta_l2_norm,
    )


def make_current_shared_adapter_state_payload(
    *,
    manifest: ModelManifestPayload,
    state: SharedAdapterStatePayload,
) -> CurrentSharedAdapterStatePayload:
    """서버 current manifest/state pair payload를 만든다."""
    return CurrentSharedAdapterStatePayload(
        schema_version=CURRENT_SHARED_ADAPTER_STATE_V1,
        manifest=manifest,
        state=state,
    )
