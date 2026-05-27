"""Legacy LoRA-classifier FedAvg projection import path."""

# ruff: noqa: F401,E501

from methods.adaptation.peft_text_classifier.aggregation.peft_encoder_fedavg_projection import (
    CLASSIFIER_HEAD_ARTIFACT_SLOT,
    LORA_ADAPTER_ARTIFACT_SLOT,
    LoraClassifierFedAvgResult,
    LoraClassifierFedAvgUpdate,
    aggregate_lora_classifier_fedavg,
    compute_lora_classifier_fedavg,
    validate_lora_classifier_update_matches_base,
)
