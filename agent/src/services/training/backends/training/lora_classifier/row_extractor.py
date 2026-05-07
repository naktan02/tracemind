"""Agent-local raw-text row extraction for LoRA-classifier training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agent.src.services.training.backends.training.base import AcceptedTrainingExample

from .config import LoraClassifierTrainingBackendConfig


@dataclass(frozen=True, slots=True)
class LoraClassifierTrainingRow:
    """Payload 생성 전에만 쓰는 agent-local raw-text 학습 row."""

    text: str
    label: str
    confidence: float
    margin: float


def build_lora_classifier_training_row(
    *,
    example: AcceptedTrainingExample,
    config: LoraClassifierTrainingBackendConfig,
) -> LoraClassifierTrainingRow:
    candidate = example.candidate
    if candidate is None:
        raise ValueError("Accepted example must carry a pseudo-label candidate.")
    text = extract_lora_classifier_training_text(example=example, config=config)
    label = candidate.label.strip()
    if not label:
        raise ValueError("Pseudo-label candidate label must not be empty.")
    return LoraClassifierTrainingRow(
        text=text,
        label=label,
        confidence=float(candidate.confidence),
        margin=float(candidate.margin),
    )


def extract_lora_classifier_training_text(
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
        "lora_classifier_trainer requires raw text or translated text on accepted "
        "examples. The fixed-embedding-only training path cannot produce LoRA "
        "classifier updates."
    )
