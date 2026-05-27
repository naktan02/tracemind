"""PEFT-backed classifier agent runtime update assembly."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from methods.adaptation.local_update_backend import AcceptedTrainingExample
from methods.adaptation.peft_text_classifier.update.local_update import (
    LoraClassifierTrainArtifacts,
    LoraClassifierTrainExecutor,
    LoraClassifierTrainingRow,
    build_peft_encoder_delta_from_rows,
    resolve_peft_encoder_label_schema,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierDelta,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierDelta,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask

from ..config import LoraClassifierTrainingBackendConfig


def build_peft_encoder_delta_update(
    *,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    accepted_examples: tuple[AcceptedTrainingExample, ...],
    config: LoraClassifierTrainingBackendConfig,
    created_at: datetime,
    train_executor: LoraClassifierTrainExecutor | None = None,
) -> LoraClassifierDelta | PeftClassifierDelta:
    rows = tuple(
        build_peft_encoder_training_row(example=example, config=config)
        for example in accepted_examples
    )
    if not rows:
        raise ValueError("accepted_examples must not be empty.")

    label_schema = resolve_peft_encoder_label_schema(
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

    return build_peft_encoder_delta_from_rows(
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
    base_artifact_ref = build_peft_encoder_base_artifact_ref(
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


def build_peft_encoder_training_row(
    *,
    example: AcceptedTrainingExample,
    config: LoraClassifierTrainingBackendConfig,
) -> LoraClassifierTrainingRow:
    candidate = example.candidate
    if candidate is None:
        raise ValueError("Accepted example must carry a pseudo-label candidate.")
    text = extract_peft_encoder_training_text(example=example, config=config)
    label = candidate.label.strip()
    if not label:
        raise ValueError("Pseudo-label candidate label must not be empty.")
    return LoraClassifierTrainingRow(
        text=text,
        label=label,
        confidence=float(candidate.confidence),
        margin=float(candidate.margin),
    )


def extract_peft_encoder_training_text(
    *,
    example: AcceptedTrainingExample,
    config: LoraClassifierTrainingBackendConfig,
) -> str:
    metadata = getattr(example, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in config.text_metadata_keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    translated_text = getattr(example.update_scored_event, "translated_text", None)
    if isinstance(translated_text, str) and translated_text.strip():
        return translated_text.strip()

    raise ValueError(
        "PEFT-backed classifier trainer requires raw text or translated text on "
        "accepted examples. The fixed-embedding-only training path cannot produce "
        "classifier adapter updates."
    )


def build_peft_encoder_base_artifact_ref(
    *,
    prefix: str,
    training_task: TrainingTask,
    created_at: datetime,
) -> str:
    timestamp = utc_timestamp(created_at)
    return "/".join(
        (
            prefix.rstrip("/"),
            slug_artifact_ref_part(training_task.round_id),
            slug_artifact_ref_part(training_task.task_id),
            timestamp,
        )
    )


def utc_timestamp(value: datetime) -> str:
    effective = (
        value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    )
    return effective.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def slug_artifact_ref_part(value: str) -> str:
    normalized = value.strip().replace("/", "_")
    if not normalized:
        raise ValueError("artifact ref path parts must not be empty.")
    return normalized
