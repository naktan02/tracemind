"""Compatibility shim for legacy lora_classifier batching imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.training.batching import (
    move_tensor_batch_to_device,
    next_cycling_batch,
)
