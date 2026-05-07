"""LoRA-classifier label schema resolution."""

from __future__ import annotations

from collections.abc import Sequence

from agent.src.services.training.backends.training.base import AcceptedTrainingExample

from .row_extractor import LoraClassifierTrainingRow


def resolve_lora_classifier_label_schema(
    *,
    accepted_examples: Sequence[AcceptedTrainingExample],
    rows: Sequence[LoraClassifierTrainingRow],
    configured_labels: Sequence[str],
) -> tuple[str, ...]:
    labels = tuple(configured_labels) or tuple(
        sorted(
            {
                label
                for example in accepted_examples
                for label in example.update_scored_event.category_scores
                if str(label).strip()
            }
        )
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
