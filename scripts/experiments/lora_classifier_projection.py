"""Compatibility shim for PEFT-backed classifier projection artifacts."""

from __future__ import annotations

from methods.adaptation.peft_text_classifier import (
    projection_artifacts as _projection_artifacts,
)

write_lora_classifier_projection_artifacts = (
    _projection_artifacts.write_lora_classifier_projection_artifacts
)
