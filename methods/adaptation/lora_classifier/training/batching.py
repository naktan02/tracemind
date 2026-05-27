"""Compatibility shim for legacy lora_classifier batching imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_text_classifier.training.batching import (
    move_tensor_batch_to_device,
    next_cycling_batch,
)
