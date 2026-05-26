"""Compatibility shim for legacy lora_classifier partitioned delta imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.update.partitioned_delta import (
    LoraClassifierPartitionDelta,
    normalize_partition_deltas,
)
