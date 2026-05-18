"""Compatibility shim for LoRA-classifier simulation inline delta imports."""

from __future__ import annotations

from methods.adaptation.lora_classifier.update import (
    simulation_inline_delta as _core,
)

SimulationInlineLoraClassifierTrainExecutor = (
    _core.SimulationInlineLoraClassifierTrainExecutor
)
