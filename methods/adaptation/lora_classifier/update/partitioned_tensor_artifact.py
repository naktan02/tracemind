"""Compatibility shim for legacy lora_classifier partitioned tensor artifact imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.update.partitioned_tensor_artifact import (
    PARTITIONED_DELTA_TENSOR_ARTIFACT_FORMAT,
    PARTITIONED_DELTA_TENSOR_ARTIFACT_INDEX_METADATA_KEY,
    PARTITIONED_DELTA_TENSOR_ARTIFACT_SCHEMA_VERSION,
    build_partitioned_delta_tensor_artifact,
    parse_partitioned_delta_tensor_artifact,
)
