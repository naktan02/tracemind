"""PEFT-encoder deterministic inline delta executor for simulation smoke."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from methods.adaptation.peft_text_encoder.training.delta_extraction import (
    peft_encoder_delta_l2_norm,
)
from methods.adaptation.peft_text_encoder.update.local_update import (
    PeftEncoderTrainArtifacts,
    PeftEncoderTrainingRow,
    PeftEncoderUpdateConfig,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask


@dataclass(frozen=True, slots=True)
class SimulationInlinePeftEncoderTrainExecutor:
    """서버 집계 가능한 deterministic PEFT encoder/classifier inline delta를 만든다."""

    peft_adapter_delta_scale: float = 0.05
    classifier_delta_scale: float = 1.0

    def train(
        self,
        *,
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        rows: Sequence[PeftEncoderTrainingRow],
        label_schema: tuple[str, ...],
        config: PeftEncoderUpdateConfig,
        created_at: datetime,
    ) -> PeftEncoderTrainArtifacts:
        del model_manifest, created_at
        if not rows:
            raise ValueError("PEFT encoder simulation inline delta requires rows.")

        peft_parameter_deltas = _build_peft_parameter_deltas(
            rows=rows,
            config=config,
            scale=self.peft_adapter_delta_scale,
        )
        classifier_head_weight_deltas = _build_classifier_head_weight_deltas(
            rows=rows,
            label_schema=label_schema,
            learning_rate=float(training_task.learning_rate),
            scale=self.classifier_delta_scale,
        )
        classifier_head_bias_deltas = _build_classifier_head_bias_deltas(
            rows=rows,
            label_schema=label_schema,
            learning_rate=float(training_task.learning_rate),
            scale=self.classifier_delta_scale,
        )
        return PeftEncoderTrainArtifacts(
            peft_parameter_deltas=peft_parameter_deltas,
            classifier_head_weight_deltas=classifier_head_weight_deltas,
            classifier_head_bias_deltas=classifier_head_bias_deltas,
            delta_l2_norm=peft_encoder_delta_l2_norm(
                peft_parameter_deltas=peft_parameter_deltas,
                classifier_head_weight_deltas=classifier_head_weight_deltas,
                classifier_head_bias_deltas=classifier_head_bias_deltas,
            ),
        )


def _build_peft_parameter_deltas(
    *,
    rows: Sequence[PeftEncoderTrainingRow],
    config: PeftEncoderUpdateConfig,
    scale: float,
) -> dict[str, list[float]]:
    rank = int(getattr(config, "rank", 1))
    vector_dim = max(1, rank)
    adapter_name = str(getattr(config, "peft_adapter_name", "lora")).strip() or "lora"
    return {
        f"{adapter_name}.simulation_adapter_a": _average_text_signature(
            rows=rows,
            vector_dim=vector_dim,
            salt="A",
            scale=scale,
        ),
        f"{adapter_name}.simulation_adapter_b": _average_text_signature(
            rows=rows,
            vector_dim=vector_dim,
            salt="B",
            scale=scale,
        ),
    }


def _average_text_signature(
    *,
    rows: Sequence[PeftEncoderTrainingRow],
    vector_dim: int,
    salt: str,
    scale: float,
) -> list[float]:
    totals = [0.0 for _ in range(vector_dim)]
    for row in rows:
        digest = hashlib.sha256(f"{salt}\n{row.label}\n{row.text}".encode()).digest()
        confidence = max(0.0, min(1.0, float(row.confidence)))
        for index in range(vector_dim):
            signed = (digest[index % len(digest)] / 255.0) - 0.5
            totals[index] += signed * confidence * scale
    return [value / len(rows) for value in totals]


def _build_classifier_head_weight_deltas(
    *,
    rows: Sequence[PeftEncoderTrainingRow],
    label_schema: tuple[str, ...],
    learning_rate: float,
    scale: float,
) -> dict[str, list[float]]:
    labels = tuple(label_schema)
    label_to_index = {label: index for index, label in enumerate(labels)}
    step_scale = _effective_step_scale(learning_rate=learning_rate, scale=scale)
    deltas = {label: [0.0 for _ in labels] for label in labels}
    for row in rows:
        row_index = label_to_index[row.label]
        confidence = max(0.0, min(1.0, float(row.confidence)))
        for label, vector in deltas.items():
            label_index = label_to_index[label]
            if label_index == row_index:
                vector[row_index] += step_scale * confidence
            else:
                vector[row_index] -= step_scale * confidence / max(1, len(labels) - 1)
    return {
        label: [value / len(rows) for value in vector]
        for label, vector in deltas.items()
    }


def _build_classifier_head_bias_deltas(
    *,
    rows: Sequence[PeftEncoderTrainingRow],
    label_schema: tuple[str, ...],
    learning_rate: float,
    scale: float,
) -> dict[str, float]:
    labels = tuple(label_schema)
    step_scale = _effective_step_scale(learning_rate=learning_rate, scale=scale)
    deltas = {label: 0.0 for label in labels}
    for row in rows:
        confidence = max(0.0, min(1.0, float(row.confidence)))
        for label in labels:
            if label == row.label:
                deltas[label] += step_scale * confidence
            else:
                deltas[label] -= step_scale * confidence / max(1, len(labels) - 1)
    return {label: value / len(rows) for label, value in deltas.items()}


def _effective_step_scale(*, learning_rate: float, scale: float) -> float:
    return max(1e-4, abs(float(learning_rate))) * float(scale)
