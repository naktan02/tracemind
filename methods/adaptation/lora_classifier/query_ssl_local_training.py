"""Compatibility shim for Query SSL LoRA local training imports."""

from __future__ import annotations

from methods.adaptation.lora_classifier.training import (
    query_ssl_local_training as _core,
)

LoraClassifierTrainerRuntimeConfig = _core.LoraClassifierTrainerRuntimeConfig
QuerySslLoraClientTrainingResult = _core.QuerySslLoraClientTrainingResult
QuerySslLoraDeltaMaterialization = _core.QuerySslLoraDeltaMaterialization
QuerySslLoraDeltaMaterializer = _core.QuerySslLoraDeltaMaterializer
QuerySslLoraObjectiveRuntimeConfig = _core.QuerySslLoraObjectiveRuntimeConfig
run_query_ssl_lora_classifier_training_core = (
    _core.run_query_ssl_lora_classifier_training_core
)
