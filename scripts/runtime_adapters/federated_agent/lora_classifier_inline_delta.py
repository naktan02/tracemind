"""FL simulation용 LoRA-classifier inline delta executor."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from methods.adaptation.lora_classifier.local_update import (
    LoraClassifierTrainArtifacts,
    LoraClassifierTrainingRow,
    LoraClassifierUpdateConfig,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask


@dataclass(frozen=True, slots=True)
class SimulationInlineLoraClassifierTrainExecutor:
    """simulation에서 서버 집계 가능한 deterministic inline delta를 만든다.

    실제 PEFT optimizer step은 이 port 뒤에 별도 executor로 붙인다. 이 executor는
    LoRA-classifier FedAvg 경로를 1-round 이상 검증하기 위한 server-materializable
    payload를 만든다.
    """

    lora_delta_scale: float = 0.05
    classifier_delta_scale: float = 1.0

    def train(
        self,
        *,
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        rows: Sequence[LoraClassifierTrainingRow],
        label_schema: tuple[str, ...],
        config: LoraClassifierUpdateConfig,
        created_at: datetime,
    ) -> LoraClassifierTrainArtifacts:
        del model_manifest, created_at
        if not rows:
            raise ValueError("LoRA-classifier simulation inline delta requires rows.")

        lora_parameter_deltas = _build_lora_parameter_deltas(
            rows=rows,
            config=config,
            scale=self.lora_delta_scale,
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
        return LoraClassifierTrainArtifacts(
            lora_parameter_deltas=lora_parameter_deltas,
            classifier_head_weight_deltas=classifier_head_weight_deltas,
            classifier_head_bias_deltas=classifier_head_bias_deltas,
            delta_l2_norm=_l2_norm(
                lora_parameter_deltas=lora_parameter_deltas,
                classifier_head_weight_deltas=classifier_head_weight_deltas,
                classifier_head_bias_deltas=classifier_head_bias_deltas,
            ),
        )


def _build_lora_parameter_deltas(
    *,
    rows: Sequence[LoraClassifierTrainingRow],
    config: LoraClassifierUpdateConfig,
    scale: float,
) -> dict[str, list[float]]:
    rank = int(getattr(config, "rank", 1))
    vector_dim = max(1, rank)
    adapter_name = str(getattr(config, "peft_adapter_name", "lora")).strip() or "lora"
    return {
        f"{adapter_name}.simulation_lora_A": _average_text_signature(
            rows=rows,
            vector_dim=vector_dim,
            salt="A",
            scale=scale,
        ),
        f"{adapter_name}.simulation_lora_B": _average_text_signature(
            rows=rows,
            vector_dim=vector_dim,
            salt="B",
            scale=scale,
        ),
    }


def _average_text_signature(
    *,
    rows: Sequence[LoraClassifierTrainingRow],
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
    rows: Sequence[LoraClassifierTrainingRow],
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
    rows: Sequence[LoraClassifierTrainingRow],
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


def _l2_norm(
    *,
    lora_parameter_deltas: Mapping[str, Sequence[float]],
    classifier_head_weight_deltas: Mapping[str, Sequence[float]],
    classifier_head_bias_deltas: Mapping[str, float],
) -> float:
    squared_norm = 0.0
    for mapping in (lora_parameter_deltas, classifier_head_weight_deltas):
        squared_norm += sum(
            float(value) * float(value)
            for vector in mapping.values()
            for value in vector
        )
    squared_norm += sum(
        float(value) * float(value) for value in classifier_head_bias_deltas.values()
    )
    return math.sqrt(squared_norm)
