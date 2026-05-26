"""Compatibility shim for legacy LoRA-classifier method-owned training imports."""

# ruff: noqa: F401,E501

from methods.adaptation.text_classifier.peft_encoder.federated_ssl.method_owned_training import (
    FederatedSslMethodLocalTrainingConfig,
    LoraClassifierTrainerRuntimeConfig,
    MethodOwnedLoraClassifierTrainingCore,
    QuerySslLoraClientTrainingResult,
    QuerySslLoraDeltaMaterializer,
    QuerySslLoraObjectiveRuntimeConfig,
    resolve_method_owned_lora_classifier_training_core,
    run_method_owned_lora_classifier_training_core,
)
