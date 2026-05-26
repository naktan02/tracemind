"""Compatibility shim for legacy lora_classifier runtime compatibility imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.runtime_compatibility import (
    LoraClassifierRuntimePayloadConfig,
    require_lora_classifier_runtime_matches_objective,
)
