"""Query SSL PEFT-backed classifier local update payload/metric 조립."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from methods.adaptation.query_classifier_adaptation.local_training_budget import (
    QuerySslLocalStepPlan,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierDelta,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierDelta,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import ClientMetricKeys, TrainingTask
from shared.src.domain.services.classification_report import safe_divide

from ..config import (
    LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
    LoraClassifierTrainingBackendConfig,
)
from ..training.delta_extraction import (
    finite_float_or_none,
    lora_classifier_delta_l2_norm,
)
from .local_update import (
    LoraClassifierTrainArtifacts,
    build_peft_encoder_delta_payload_from_artifacts,
)
from .partitioned_delta import LoraClassifierPartitionDelta


@dataclass(frozen=True, slots=True)
class QuerySslLoraUpdateBuildResult:
    """FL adapter가 envelope과 summary에 쓰는 Query SSL local update 결과."""

    update_payload: LoraClassifierDelta | PeftClassifierDelta
    client_metrics: Mapping[str, float]
    accepted_unlabeled_count: int


QuerySslPeftEncoderUpdateBuildResult = QuerySslLoraUpdateBuildResult


def build_query_ssl_peft_encoder_update_payload(
    *,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    lora_config: LoraClassifierTrainingBackendConfig,
    labels: Sequence[str],
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    step_plan: QuerySslLocalStepPlan,
    history_record: Mapping[str, object],
    lora_parameter_deltas: Mapping[str, Sequence[float]],
    classifier_head_weight_deltas: Mapping[str, Sequence[float]],
    classifier_head_bias_deltas: Mapping[str, float],
    partitioned_deltas: Mapping[str, LoraClassifierPartitionDelta] | None = None,
    created_at: datetime,
    delta_format: str = LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
    lora_delta_artifact_ref: str | None = None,
    classifier_head_delta_artifact_ref: str | None = None,
    partitioned_deltas_artifact_ref: str | None = None,
    include_inline_deltas: bool = True,
) -> QuerySslLoraUpdateBuildResult:
    """학습 결과 delta와 history를 shared payload/metric으로 변환한다."""

    normalized_delta_format = str(delta_format).strip()
    if not normalized_delta_format:
        raise ValueError("delta_format must not be empty.")
    has_primary_artifact_refs = (
        lora_delta_artifact_ref is not None
        and classifier_head_delta_artifact_ref is not None
    )
    has_partitioned_artifact_ref = partitioned_deltas_artifact_ref is not None
    if not include_inline_deltas and not (
        has_primary_artifact_refs or has_partitioned_artifact_ref
    ):
        raise ValueError(
            "artifact-ref Query SSL PEFT update requires adapter/head delta refs or "
            "partitioned_deltas_artifact_ref."
        )
    util_ratio = finite_float_or_none(history_record.get("train_util_ratio"))
    accepted_unlabeled_count = int(round((util_ratio or 0.0) * len(unlabeled_rows)))
    delta_l2_norm = lora_classifier_delta_l2_norm(
        lora_parameter_deltas=lora_parameter_deltas,
        classifier_head_weight_deltas=classifier_head_weight_deltas,
        classifier_head_bias_deltas=classifier_head_bias_deltas,
    )
    update_payload = build_peft_encoder_delta_payload_from_artifacts(
        training_task=training_task,
        model_manifest=model_manifest,
        config=lora_config,
        label_schema=labels,
        example_count=len(labeled_rows) + len(unlabeled_rows),
        label_counts=_build_labeled_row_label_counts(labeled_rows),
        artifacts=LoraClassifierTrainArtifacts(
            lora_delta_artifact_ref=lora_delta_artifact_ref,
            classifier_head_delta_artifact_ref=classifier_head_delta_artifact_ref,
            lora_parameter_deltas=(
                lora_parameter_deltas if include_inline_deltas else None
            ),
            classifier_head_weight_deltas=(
                classifier_head_weight_deltas if include_inline_deltas else None
            ),
            classifier_head_bias_deltas=(
                classifier_head_bias_deltas if include_inline_deltas else None
            ),
            partitioned_deltas=(
                None
                if partitioned_deltas_artifact_ref is not None
                else partitioned_deltas
            ),
            partitioned_deltas_artifact_ref=partitioned_deltas_artifact_ref,
            delta_l2_norm=delta_l2_norm,
        ),
        delta_format=normalized_delta_format,
        mean_confidence=None,
        mean_margin=None,
        created_at=created_at,
    )
    return QuerySslLoraUpdateBuildResult(
        update_payload=update_payload,
        client_metrics=build_query_ssl_peft_encoder_client_metrics(
            update_payload=update_payload,
            step_plan=step_plan,
            history_record=history_record,
            labeled_count=len(labeled_rows),
            unlabeled_count=len(unlabeled_rows),
            accepted_unlabeled_count=accepted_unlabeled_count,
        ),
        accepted_unlabeled_count=accepted_unlabeled_count,
    )


def build_query_ssl_peft_encoder_client_metrics(
    *,
    update_payload: LoraClassifierDelta | PeftClassifierDelta,
    step_plan: QuerySslLocalStepPlan,
    history_record: Mapping[str, object],
    labeled_count: int,
    unlabeled_count: int,
    accepted_unlabeled_count: int,
) -> dict[str, float]:
    """Query SSL LoRA local update client metric을 조립한다."""

    metrics: dict[str, float] = {
        ClientMetricKeys.SELECTED_EXAMPLES: float(update_payload.example_count),
        ClientMetricKeys.ACCEPTED_RATIO: safe_divide(
            accepted_unlabeled_count,
            unlabeled_count,
        ),
        ClientMetricKeys.MEAN_CONFIDENCE: 0.0,
        ClientMetricKeys.MEAN_MARGIN: 0.0,
        ClientMetricKeys.DELTA_L2_NORM: update_payload.l2_norm(),
        "query_ssl_local_steps": float(step_plan.total_steps),
        "query_ssl_full_epoch_steps": float(step_plan.full_epoch_steps),
        "query_ssl_local_epochs": float(step_plan.local_epochs),
        "query_ssl_max_steps": float(step_plan.max_steps),
        "query_ssl_labeled_loader_steps": float(step_plan.labeled_loader_steps),
        "query_ssl_unlabeled_loader_steps": float(step_plan.unlabeled_loader_steps),
        "query_ssl_labeled_count": float(labeled_count),
        "query_ssl_unlabeled_count": float(unlabeled_count),
        "query_ssl_accepted_unlabeled_count": float(accepted_unlabeled_count),
    }
    for key, value in history_record.items():
        numeric_value = finite_float_or_none(value)
        if numeric_value is None:
            continue
        if key == "train_unsup_loss":
            metrics["unlabeled_loss"] = numeric_value
        if key == "train_sup_loss":
            metrics["supervised_loss"] = numeric_value
        if key == "train_util_ratio":
            metrics["pseudo_label_acceptance_rate"] = numeric_value
        if str(key).startswith("train_"):
            metrics[f"query_ssl_{key}"] = numeric_value
    return metrics


build_query_ssl_lora_update_payload = build_query_ssl_peft_encoder_update_payload
build_query_ssl_lora_client_metrics = build_query_ssl_peft_encoder_client_metrics


def _build_labeled_row_label_counts(
    labeled_rows: Sequence[LabeledQueryRow],
) -> dict[str, int]:
    label_counts: dict[str, int] = {}
    for row in labeled_rows:
        label = str(row["mapped_label_4"])
        label_counts[label] = label_counts.get(label, 0) + 1
    return label_counts
