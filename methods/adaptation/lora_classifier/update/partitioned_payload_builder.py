"""Compatibility shim for legacy lora_classifier partitioned payload imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_text_classifier.update.partitioned_payload_builder import (
    build_partitioned_delta_payload,
)
