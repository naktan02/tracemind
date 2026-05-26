"""Compatibility shim for legacy lora_classifier initial state imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.initial_state import (
    LoraClassifierInitialStateConfig,
    build_initial_lora_classifier_state,
)
