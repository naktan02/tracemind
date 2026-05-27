"""Compatibility shim for legacy lora_classifier scalar metric imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_text_classifier.training.scalar_metrics import (
    ScalarMetricAccumulator,
    tensor_mapping_l2,
)
