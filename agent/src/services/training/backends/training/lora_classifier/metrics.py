"""LoRA-classifier client metric extraction."""

from __future__ import annotations

from shared.src.contracts.adapter_contracts import LoraClassifierDelta
from shared.src.contracts.training_contracts import ClientMetricKeys
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)


def build_lora_classifier_client_metrics(
    update: SharedAdapterUpdate,
) -> dict[str, float]:
    if not isinstance(update, LoraClassifierDelta):
        raise TypeError(
            "LoraClassifierTrainingBackend expects LoraClassifierDelta "
            f"for metric extraction, got {type(update)!r}."
        )
    return {
        ClientMetricKeys.MEAN_CONFIDENCE: update.mean_confidence or 0.0,
        ClientMetricKeys.MEAN_MARGIN: update.mean_margin or 0.0,
        ClientMetricKeys.DELTA_L2_NORM: update.l2_norm(),
        "lora_training_rows": float(update.example_count),
        "lora_label_schema_size": float(len(update.label_schema)),
    }
