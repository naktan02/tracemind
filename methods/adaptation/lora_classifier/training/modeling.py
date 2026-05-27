"""Compatibility shim for legacy lora_classifier modeling imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_text_classifier.training.modeling import (
    LoraClassifierModelRuntimeConfig,
    LoraTextClassifier,
    build_lora_text_classifier_from_config,
    build_model,
    count_parameters,
    require_transformer_stack,
)
