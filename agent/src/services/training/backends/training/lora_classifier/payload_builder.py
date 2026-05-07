"""LoRA-classifier update payload construction."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from agent.src.services.training.backends.training.base import AcceptedTrainingExample
from shared.src.contracts.adapter_contracts import (
    LoraClassifierDelta,
    make_lora_classifier_delta_payload,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask

from .artifact_refs import build_lora_classifier_base_artifact_ref
from .config import LoraClassifierTrainingBackendConfig
from .label_schema import resolve_lora_classifier_label_schema
from .row_extractor import build_lora_classifier_training_row
from .train_executor import (
    LoraClassifierTrainArtifacts,
    LoraClassifierTrainExecutor,
)


def build_lora_classifier_delta_update(
    *,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    accepted_examples: tuple[AcceptedTrainingExample, ...],
    config: LoraClassifierTrainingBackendConfig,
    created_at: datetime,
    train_executor: LoraClassifierTrainExecutor | None = None,
) -> LoraClassifierDelta:
    rows = tuple(
        build_lora_classifier_training_row(example=example, config=config)
        for example in accepted_examples
    )
    if not rows:
        raise ValueError("accepted_examples must not be empty.")

    label_schema = resolve_lora_classifier_label_schema(
        accepted_examples=accepted_examples,
        rows=rows,
        configured_labels=config.label_schema,
    )
    label_counts = Counter(row.label for row in rows)
    artifacts = (
        train_executor.train(
            training_task=training_task,
            model_manifest=model_manifest,
            rows=rows,
            label_schema=label_schema,
            config=config,
            created_at=created_at,
        )
        if train_executor is not None
        else _build_payload_only_artifacts(
            training_task=training_task,
            config=config,
            created_at=created_at,
        )
    )

    return make_lora_classifier_delta_payload(
        model_id=model_manifest.model_id,
        base_model_revision=model_manifest.model_revision,
        training_scope=training_task.training_scope,
        backbone=config.to_backbone_payload(),
        lora_config=config.to_lora_config_payload(),
        label_schema=label_schema,
        example_count=len(rows),
        lora_delta_artifact_ref=artifacts.lora_delta_artifact_ref,
        classifier_head_delta_artifact_ref=(
            artifacts.classifier_head_delta_artifact_ref
        ),
        delta_format=config.delta_format,
        mean_confidence=sum(row.confidence for row in rows) / len(rows),
        mean_margin=sum(row.margin for row in rows) / len(rows),
        label_counts=dict(sorted(label_counts.items())),
        delta_l2_norm=artifacts.delta_l2_norm,
        created_at=created_at,
    )


def _build_payload_only_artifacts(
    *,
    training_task: TrainingTask,
    config: LoraClassifierTrainingBackendConfig,
    created_at: datetime,
) -> LoraClassifierTrainArtifacts:
    base_artifact_ref = build_lora_classifier_base_artifact_ref(
        prefix=config.artifact_ref_prefix,
        training_task=training_task,
        created_at=created_at,
    )
    return LoraClassifierTrainArtifacts(
        lora_delta_artifact_ref=f"{base_artifact_ref}/lora_delta",
        classifier_head_delta_artifact_ref=(
            f"{base_artifact_ref}/classifier_head_delta"
        ),
        delta_l2_norm=0.0,
    )
