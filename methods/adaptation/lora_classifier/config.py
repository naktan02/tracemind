"""Compatibility shim for legacy lora_classifier config imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_text_classifier.config import (
    LORA_CLASSIFIER_DELTA_FORMAT_AGENT_LOCAL,
    LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
    LORA_CLASSIFIER_DELTA_FORMAT_SERVER_UPLOADED,
    LORA_CLASSIFIER_FAMILY_EXTRA_SCOPE,
    LORA_CLASSIFIER_TRAINING_BACKEND_EXTRA_SCOPE,
    LORA_CLASSIFIER_TRAINING_BACKEND_NAME,
    LoraClassifierTrainingBackendConfig,
    build_lora_classifier_training_backend_config,
)
