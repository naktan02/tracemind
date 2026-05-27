"""Compatibility shim for legacy lora_classifier optimizer step imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_text_classifier.training.optimizer_step import (
    run_optimizer_loss_step,
)
