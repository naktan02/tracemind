"""Compatibility shim for legacy lora_classifier inline delta imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.update.simulation_inline_delta import (
    SimulationInlineLoraClassifierTrainExecutor,
)
