"""Compatibility shim for legacy lora_classifier scalar metric imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.training.scalar_metrics import (
    ScalarMetricAccumulator,
    tensor_mapping_l2,
)
