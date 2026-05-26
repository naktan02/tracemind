"""Compatibility shim for legacy lora_classifier merged tensor artifact imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.update.merged_tensor_artifact import (
    HEAD_DELTA_TENSOR_ARTIFACT_INDEX_METADATA_KEY,
    MERGED_DELTA_TENSOR_ARTIFACT_FORMAT,
    MERGED_HEAD_DELTA_TENSOR_ARTIFACT_SCHEMA_VERSION,
    MERGED_LORA_DELTA_TENSOR_ARTIFACT_SCHEMA_VERSION,
    LORA_DELTA_TENSOR_ARTIFACT_INDEX_METADATA_KEY,
    build_classifier_head_delta_tensor_artifact,
    build_lora_delta_tensor_artifact,
    parse_classifier_head_delta_tensor_artifact,
    parse_lora_delta_tensor_artifact,
)
