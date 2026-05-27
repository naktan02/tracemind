"""Compatibility shim for legacy lora_classifier JSON artifact imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_text_classifier.update.json_delta_artifact import (
    HEAD_DELTA_ARTIFACT_SCHEMA_VERSION,
    LORA_DELTA_ARTIFACT_SCHEMA_VERSION,
    PARTITIONED_DELTA_ARTIFACT_SCHEMA_VERSION,
    build_classifier_head_delta_json_artifact_payload,
    build_lora_delta_json_artifact_payload,
    build_partitioned_delta_json_artifact_payload,
)
