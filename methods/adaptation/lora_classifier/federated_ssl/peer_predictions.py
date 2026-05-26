"""Compatibility shim for legacy LoRA-classifier peer prediction imports."""

# ruff: noqa: F401,E501

from methods.adaptation.text_classifier.peft_encoder.federated_ssl.peer_predictions import (
    LORA_CLASSIFIER_PEER_SNAPSHOT_KIND,
    LoraClassifierHelperWeakProbabilityProvider,
    LoraClassifierTrainerRuntimeConfig,
    build_lora_classifier_helper_probability_provider,
    build_lora_classifier_peer_client_snapshot,
    compute_lora_classifier_probe_vector,
    extract_lora_classifier_materialized_state,
)
