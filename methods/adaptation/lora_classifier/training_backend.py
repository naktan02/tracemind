"""Compatibility shim for legacy lora_classifier training backend imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_text_classifier.training_backend import (
    LORA_CLASSIFIER_TRAINING_BACKEND_CATALOG_ENTRY,
    LoraClassifierTrainingBackend,
    build_lora_classifier_client_metrics,
    build_lora_classifier_training_backend,
)
