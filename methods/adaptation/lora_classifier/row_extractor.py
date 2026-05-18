"""Compatibility shim for LoRA-classifier row extraction imports."""

from __future__ import annotations

from methods.adaptation.lora_classifier.update.row_extractor import (
    build_lora_classifier_training_row as build_lora_classifier_training_row,
)
from methods.adaptation.lora_classifier.update.row_extractor import (
    extract_lora_classifier_training_text as extract_lora_classifier_training_text,
)
