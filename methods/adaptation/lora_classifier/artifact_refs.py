"""Compatibility shim for LoRA-classifier artifact ref imports."""

from __future__ import annotations

from methods.adaptation.lora_classifier.update.artifact_refs import (
    build_lora_classifier_base_artifact_ref as build_lora_classifier_base_artifact_ref,
)
from methods.adaptation.lora_classifier.update.artifact_refs import (
    slug_artifact_ref_part as slug_artifact_ref_part,
)
from methods.adaptation.lora_classifier.update.artifact_refs import (
    utc_timestamp as utc_timestamp,
)
