"""LoRA-classifier local update core."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from shared.src.contracts.adapter_contract_families.factories import (
    make_lora_classifier_delta_payload,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierDelta,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask


@dataclass(frozen=True, slots=True)
class LoraClassifierTrainingRow:
    """LoRA-classifier update core가 소비하는 raw-text 학습 row."""

    text: str
    label: str
    confidence: float
    margin: float


@dataclass(frozen=True, slots=True)
class LoraClassifierTrainArtifacts:
    """LoRA train executor가 payload core에 돌려주는 artifact snapshot."""

    lora_delta_artifact_ref: str | None = None
    classifier_head_delta_artifact_ref: str | None = None
    lora_parameter_deltas: Mapping[str, Sequence[float]] | None = None
    classifier_head_weight_deltas: Mapping[str, Sequence[float]] | None = None
    classifier_head_bias_deltas: Mapping[str, float] | None = None
    delta_l2_norm: float = 0.0


class LoraClassifierUpdateConfig(Protocol):
    """Payload core가 필요한 LoRA-classifier config surface."""

    delta_format: str

    def to_backbone_payload(self) -> Mapping[str, str | int]:
        """Shared payload에 기록할 backbone/tokenizer snapshot을 반환한다."""

    def to_lora_config_payload(self) -> Mapping[str, str | int | float | bool]:
        """Shared payload에 기록할 LoRA config snapshot을 반환한다."""


class LoraClassifierTrainExecutor(Protocol):
    """실제 PEFT/LoRA 학습 실행기가 만족해야 하는 boundary."""

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
        """agent-local raw text rows로 LoRA/classifier delta artifact를 만든다."""


class NotImplementedLoraClassifierTrainExecutor:
    """실제 LoRA train step이 붙기 전 명시적으로 실패하는 executor."""

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
        del training_task, model_manifest, rows, label_schema, config, created_at
        raise NotImplementedError(
            "LoRA-classifier train executor is not implemented yet. "
            "Use the payload-only backend path until PEFT artifact materialization "
            "is wired."
        )


def resolve_lora_classifier_label_schema(
    *,
    rows: Sequence[LoraClassifierTrainingRow],
    configured_labels: Sequence[str],
    candidate_label_space: Sequence[str] = (),
) -> tuple[str, ...]:
    """설정값, 후보 label space, row label 순서로 payload label schema를 결정한다."""

    labels = tuple(configured_labels) or tuple(
        sorted({label for label in candidate_label_space if str(label).strip()})
    )
    if not labels:
        labels = tuple(sorted({row.label for row in rows}))

    missing_labels = sorted({row.label for row in rows} - set(labels))
    if missing_labels:
        raise ValueError(
            "LoRA-classifier label_schema must include accepted labels: "
            f"{missing_labels}."
        )
    return labels


def build_lora_classifier_delta_from_rows(
    *,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    rows: Sequence[LoraClassifierTrainingRow],
    label_schema: tuple[str, ...],
    config: LoraClassifierUpdateConfig,
    artifacts: LoraClassifierTrainArtifacts,
    created_at: datetime,
) -> LoraClassifierDelta:
    """Resolved rows와 artifact snapshot으로 shared delta payload를 만든다."""

    if not rows:
        raise ValueError("rows must not be empty.")

    label_counts = Counter(row.label for row in rows)
    return build_lora_classifier_delta_payload_from_artifacts(
        training_task=training_task,
        model_manifest=model_manifest,
        config=config,
        label_schema=label_schema,
        example_count=len(rows),
        label_counts=label_counts,
        artifacts=artifacts,
        delta_format=config.delta_format,
        mean_confidence=sum(row.confidence for row in rows) / len(rows),
        mean_margin=sum(row.margin for row in rows) / len(rows),
        created_at=created_at,
    )


def build_lora_classifier_delta_payload_from_artifacts(
    *,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    config: LoraClassifierUpdateConfig,
    label_schema: Sequence[str],
    example_count: int,
    label_counts: Mapping[str, int],
    artifacts: LoraClassifierTrainArtifacts,
    delta_format: str,
    mean_confidence: float | None,
    mean_margin: float | None,
    created_at: datetime,
) -> LoraClassifierDelta:
    """LoRA/head artifact snapshot을 shared delta payload로 정규화한다."""

    if example_count <= 0:
        raise ValueError("example_count must be positive.")
    return make_lora_classifier_delta_payload(
        model_id=model_manifest.model_id,
        base_model_revision=model_manifest.model_revision,
        training_scope=training_task.training_scope,
        backbone=dict(config.to_backbone_payload()),
        lora_config=dict(config.to_lora_config_payload()),
        label_schema=tuple(str(label) for label in label_schema),
        example_count=example_count,
        lora_delta_artifact_ref=artifacts.lora_delta_artifact_ref,
        classifier_head_delta_artifact_ref=(
            artifacts.classifier_head_delta_artifact_ref
        ),
        lora_parameter_deltas=(
            None
            if artifacts.lora_parameter_deltas is None
            else {
                key: [float(value) for value in values]
                for key, values in artifacts.lora_parameter_deltas.items()
            }
        ),
        classifier_head_weight_deltas=(
            None
            if artifacts.classifier_head_weight_deltas is None
            else {
                key: [float(value) for value in values]
                for key, values in artifacts.classifier_head_weight_deltas.items()
            }
        ),
        classifier_head_bias_deltas=(
            {}
            if artifacts.classifier_head_bias_deltas is None
            else {
                key: float(value)
                for key, value in artifacts.classifier_head_bias_deltas.items()
            }
        ),
        delta_format=delta_format,
        mean_confidence=mean_confidence,
        mean_margin=mean_margin,
        label_counts={
            str(label): int(count) for label, count in sorted(label_counts.items())
        },
        delta_l2_norm=artifacts.delta_l2_norm,
        created_at=created_at,
    )
