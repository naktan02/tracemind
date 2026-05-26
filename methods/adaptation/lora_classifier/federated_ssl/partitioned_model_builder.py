"""Compatibility shim for legacy partitioned model builder imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.federated_ssl.partitioned.model_builder import (
    LoraClassifierPartitionRuntimeConfig,
    LoraTextClassifierFactory,
    PartitionedLoraTextClassifierBuildResult,
    build_partitioned_lora_text_classifier_from_config,
)
