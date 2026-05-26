"""Compatibility shim for legacy lora_classifier partitioned payload imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.update.partitioned_payload_builder import (
    build_partitioned_delta_payload,
)
