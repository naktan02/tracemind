"""Compatibility shim for legacy lora_classifier Query SSL local training imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_text_classifier.training.query_ssl_local_training import (
    LoraClassifierTrainerRuntimeConfig,
    QuerySslLoraClientTrainingResult,
    QuerySslLoraDeltaMaterialization,
    QuerySslLoraDeltaMaterializer,
    QuerySslLoraObjectiveRuntimeConfig,
    run_query_ssl_lora_classifier_training_core,
)
