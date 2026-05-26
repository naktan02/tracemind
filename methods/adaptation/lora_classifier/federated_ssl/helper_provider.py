"""Compatibility shim for legacy LoRA-classifier helper provider imports."""

# ruff: noqa: F401,E501

from methods.adaptation.text_classifier.peft_encoder.federated_ssl.helper_provider import (
    LoraClassifierTrainerRuntimeConfig,
    build_lora_classifier_helper_provider_for_local_ssl_policy,
)
