"""Compatibility shim for legacy lora_classifier partitioned delta imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_text_classifier.training.partitioned_deltas import (
    AdapterClassifierDeltaBundle,
    build_adapter_classifier_delta_bundle,
    build_lora_classifier_partition_delta_from_parameter_deltas,
    diff_parameter_snapshots,
    named_trainable_parameter_tensors,
    project_adapter_classifier_delta_bundle_to_lora_partition_delta,
    snapshot_trainable_parameter_tensors,
)
