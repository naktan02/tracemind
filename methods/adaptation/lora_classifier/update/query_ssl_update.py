"""Query SSL LoRA-classifier local update payload/metric 조립."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from methods.adaptation.query_classifier_adaptation.local_training_budget import (
    QuerySslLocalStepPlan,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_lora_classifier_delta_payload,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierDelta,
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


@dataclass(frozen=True, slots=True)
class QuerySslLoraUpdateBuildResult:
    """FL adapter가 envelope과 summary에 쓰는 Query SSL local update 결과."""

    update_payload: LoraClassifierDelta
    client_metrics: Mapping[str, float]
    accepted_unlabeled_count: int


def build_query_ssl_lora_update_payload(
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
    created_at: datetime,
    delta_format: str = LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
    lora_delta_artifact_ref: str | None = None,
    classifier_head_delta_artifact_ref: str | None = None,
    include_inline_deltas: bool = True,
) -> QuerySslLoraUpdateBuildResult:
    """학습 결과 delta와 history를 shared payload/metric으로 변환한다."""

    normalized_delta_format = str(delta_format).strip()
    if not normalized_delta_format:
        raise ValueError("delta_format must not be empty.")
    if not include_inline_deltas and (
        lora_delta_artifact_ref is None or classifier_head_delta_artifact_ref is None
    ):
        raise ValueError(
            "artifact-ref Query SSL LoRA update requires both lora/head delta refs."
        )
    util_ratio = finite_float_or_none(history_record.get("train_util_ratio"))
    accepted_unlabeled_count = int(round((util_ratio or 0.0) * len(unlabeled_rows)))
    delta_l2_norm = lora_classifier_delta_l2_norm(
        lora_parameter_deltas=lora_parameter_deltas,
        classifier_head_weight_deltas=classifier_head_weight_deltas,
        classifier_head_bias_deltas=classifier_head_bias_deltas,
    )
    update_payload = make_lora_classifier_delta_payload(
        model_id=model_manifest.model_id,
        base_model_revision=model_manifest.model_revision,
        training_scope=training_task.training_scope,
        backbone=lora_config.to_backbone_payload(),
        lora_config=lora_config.to_lora_config_payload(),
        label_schema=tuple(str(label) for label in labels),
        example_count=len(labeled_rows) + len(unlabeled_rows),
        lora_delta_artifact_ref=lora_delta_artifact_ref,
        classifier_head_delta_artifact_ref=classifier_head_delta_artifact_ref,
        lora_parameter_deltas=(
            {
                str(key): [float(value) for value in values]
                for key, values in lora_parameter_deltas.items()
            }
            if include_inline_deltas
            else None
        ),
        classifier_head_weight_deltas=(
            {
                str(key): [float(value) for value in values]
                for key, values in classifier_head_weight_deltas.items()
            }
            if include_inline_deltas
            else None
        ),
        classifier_head_bias_deltas={
            str(key): float(value) for key, value in classifier_head_bias_deltas.items()
        }
        if include_inline_deltas
        else {},
        delta_format=normalized_delta_format,
        mean_confidence=None,
        mean_margin=None,
        label_counts=dict(
            sorted(Counter(row["mapped_label_4"] for row in labeled_rows).items())
        ),
        delta_l2_norm=delta_l2_norm,
        created_at=created_at,
    )
    return QuerySslLoraUpdateBuildResult(
        update_payload=update_payload,
        client_metrics=build_query_ssl_lora_client_metrics(
            update_payload=update_payload,
            step_plan=step_plan,
            history_record=history_record,
            labeled_count=len(labeled_rows),
            unlabeled_count=len(unlabeled_rows),
            accepted_unlabeled_count=accepted_unlabeled_count,
        ),
        accepted_unlabeled_count=accepted_unlabeled_count,
    )


def build_query_ssl_lora_client_metrics(
    *,
    update_payload: LoraClassifierDelta,
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
