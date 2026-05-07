"""LoRA-classifier agent runtime update assembly."""

from __future__ import annotations

from datetime import datetime

from methods.adaptation.local_update_backend import AcceptedTrainingExample
from methods.adaptation.lora_classifier.local_update import (
    LoraClassifierTrainArtifacts,
    LoraClassifierTrainExecutor,
    build_lora_classifier_delta_from_rows,
    resolve_lora_classifier_label_schema,
)
from shared.src.contracts.adapter_contracts import LoraClassifierDelta
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask

from .artifact_refs import build_lora_classifier_base_artifact_ref
from .config import LoraClassifierTrainingBackendConfig
from .row_extractor import build_lora_classifier_training_row


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
        rows=rows,
        configured_labels=config.label_schema,
        candidate_label_space=_build_candidate_label_space(accepted_examples),
    )
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

    return build_lora_classifier_delta_from_rows(
        training_task=training_task,
        model_manifest=model_manifest,
        rows=rows,
        label_schema=label_schema,
        config=config,
        artifacts=artifacts,
        created_at=created_at,
    )


def _build_candidate_label_space(
    accepted_examples: tuple[AcceptedTrainingExample, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(label).strip()
                for example in accepted_examples
                for label in example.update_scored_event.category_scores
                if str(label).strip()
            }
        )
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
