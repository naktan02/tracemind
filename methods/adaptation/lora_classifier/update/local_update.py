"""Compatibility shim for legacy lora_classifier local update imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.update.local_update import (
    LoraClassifierTrainArtifacts,
    LoraClassifierTrainExecutor,
    LoraClassifierTrainingRow,
    LoraClassifierUpdateConfig,
    build_lora_classifier_delta_from_rows,
    build_lora_classifier_delta_payload_from_artifacts,
    resolve_lora_classifier_label_schema,
)
