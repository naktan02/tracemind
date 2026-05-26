"""Compatibility shim for legacy lora_classifier materialization imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.update.materialization import (
    CLASSIFIER_HEAD_STATE_BIASES_KEY,
    CLASSIFIER_HEAD_STATE_WEIGHTS_KEY,
    LORA_STATE_PARAMETERS_KEY,
    PARTITIONED_CLASSIFIER_HEAD_STATE_BIASES_KEY,
    PARTITIONED_CLASSIFIER_HEAD_STATE_WEIGHTS_KEY,
    PARTITIONED_LORA_STATE_PARAMETERS_KEY,
    LoraClassifierMaterializedState,
    LoraClassifierMaterializedUpdate,
    compact_lora_classifier_materialized_state,
    materialize_base_lora_classifier_partitioned_state,
    materialize_base_lora_classifier_state,
    materialize_lora_classifier_partitioned_update,
    materialize_lora_classifier_update,
)
