"""PEFT text encoder/head local update core."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from shared.src.contracts.adapter_contract_families.factories import (
    make_peft_classifier_delta_payload,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_ADAPTER_KIND,
    PeftClassifierDelta,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask

from .partitioned_delta import PeftEncoderPartitionDelta


@dataclass(frozen=True, slots=True)
class PeftEncoderTrainingRow:
    """PEFT encoder update core가 소비하는 raw-text 학습 row."""

    text: str
    label: str
    confidence: float
    margin: float


@dataclass(frozen=True, slots=True)
class PeftEncoderTrainArtifacts:
    """PEFT train executor가 payload core에 돌려주는 artifact snapshot."""

    peft_adapter_delta_artifact_ref: str | None = None
    classifier_head_delta_artifact_ref: str | None = None
    peft_parameter_deltas: Mapping[str, Sequence[float]] | None = None
    classifier_head_weight_deltas: Mapping[str, Sequence[float]] | None = None
    classifier_head_bias_deltas: Mapping[str, float] | None = None
    partitioned_deltas: Mapping[str, PeftEncoderPartitionDelta] | None = None
    partitioned_deltas_artifact_ref: str | None = None
    delta_l2_norm: float = 0.0


class PeftEncoderUpdateConfig(Protocol):
    """Payload core가 필요한 PEFT encoder config surface."""

    delta_format: str
    payload_adapter_kind: str

    def to_backbone_payload(self) -> Mapping[str, str | int]:
        """Shared payload에 기록할 backbone/tokenizer snapshot을 반환한다."""

    def to_peft_adapter_config_payload(self) -> Mapping[str, object]:
        """Shared payload에 기록할 PEFT adapter config snapshot을 반환한다."""


class PeftEncoderTrainExecutor(Protocol):
    """실제 PEFT 학습 실행기가 만족해야 하는 boundary."""

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
        """agent-local raw text rows로 PEFT/classifier delta artifact를 만든다."""


def resolve_peft_encoder_label_schema(
    *,
    rows: Sequence[PeftEncoderTrainingRow],
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
            "PEFT text encoder/head label_schema must include accepted labels: "
            f"{missing_labels}."
        )
    return labels


def build_peft_encoder_delta_from_rows(
    *,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    rows: Sequence[PeftEncoderTrainingRow],
    label_schema: tuple[str, ...],
    config: PeftEncoderUpdateConfig,
    artifacts: PeftEncoderTrainArtifacts,
    created_at: datetime,
) -> PeftClassifierDelta:
    """Resolved rows와 artifact snapshot으로 shared delta payload를 만든다."""

    if not rows:
        raise ValueError("rows must not be empty.")

    label_counts = Counter(row.label for row in rows)
    return build_peft_encoder_delta_payload_from_artifacts(
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


def build_peft_encoder_delta_payload_from_artifacts(
    *,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    config: PeftEncoderUpdateConfig,
    label_schema: Sequence[str],
    example_count: int,
    label_counts: Mapping[str, int],
    artifacts: PeftEncoderTrainArtifacts,
    delta_format: str,
    mean_confidence: float | None,
    mean_margin: float | None,
    created_at: datetime,
) -> PeftClassifierDelta:
    """PEFT adapter/head artifact snapshot을 shared delta payload로 정규화한다."""

    if example_count <= 0:
        raise ValueError("example_count must be positive.")
    if config.payload_adapter_kind != PEFT_CLASSIFIER_ADAPTER_KIND:
        raise ValueError(
            "PEFT text encoder/head update builder only supports "
            f"peft_classifier payloads, got {config.payload_adapter_kind!r}."
        )
    return make_peft_classifier_delta_payload(
        model_id=model_manifest.model_id,
        base_model_revision=model_manifest.model_revision,
        training_scope=training_task.training_scope,
        backbone=dict(config.to_backbone_payload()),
        peft_adapter_config=dict(config.to_peft_adapter_config_payload()),
        label_schema=tuple(str(label) for label in label_schema),
        example_count=example_count,
        peft_adapter_delta_artifact_ref=artifacts.peft_adapter_delta_artifact_ref,
        classifier_head_delta_artifact_ref=(
            artifacts.classifier_head_delta_artifact_ref
        ),
        peft_parameter_deltas=_normalize_vector_mapping(
            artifacts.peft_parameter_deltas
        ),
        classifier_head_weight_deltas=_normalize_vector_mapping(
            artifacts.classifier_head_weight_deltas
        ),
        classifier_head_bias_deltas=_normalize_scalar_mapping(
            artifacts.classifier_head_bias_deltas
        ),
        partitioned_deltas=_build_partitioned_delta_payload(
            artifacts.partitioned_deltas,
            peft_parameter_field_name="peft_parameter_deltas",
        ),
        partitioned_deltas_artifact_ref=artifacts.partitioned_deltas_artifact_ref,
        delta_format=delta_format,
        mean_confidence=mean_confidence,
        mean_margin=mean_margin,
        label_counts={
            str(label): int(count) for label, count in sorted(label_counts.items())
        },
        delta_l2_norm=artifacts.delta_l2_norm,
        created_at=created_at,
    )


def _build_partitioned_delta_payload(
    partitioned_deltas: Mapping[str, PeftEncoderPartitionDelta] | None,
    *,
    peft_parameter_field_name: str,
) -> dict[str, dict[str, object]] | None:
    if partitioned_deltas is None:
        return None
    return {
        str(name): {
            peft_parameter_field_name: {
                key: [float(value) for value in values]
                for key, values in delta.peft_parameter_deltas.items()
            },
            "classifier_head_weight_deltas": {
                key: [float(value) for value in values]
                for key, values in delta.classifier_head_weight_deltas.items()
            },
            "classifier_head_bias_deltas": {
                key: float(value)
                for key, value in delta.classifier_head_bias_deltas.items()
            },
        }
        for name, delta in sorted(partitioned_deltas.items())
    }


def _normalize_vector_mapping(
    values: Mapping[str, Sequence[float]] | None,
) -> dict[str, list[float]] | None:
    if values is None:
        return None
    return {key: [float(value) for value in vector] for key, vector in values.items()}


def _normalize_scalar_mapping(
    values: Mapping[str, float] | None,
) -> dict[str, float]:
    if values is None:
        return {}
    return {key: float(value) for key, value in values.items()}
