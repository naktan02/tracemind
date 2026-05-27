"""Compatibility shim for legacy lora_classifier partitioned delta imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_text_classifier.update.partitioned_delta import (
    LoraClassifierPartitionDelta,
    normalize_partition_deltas,
)
