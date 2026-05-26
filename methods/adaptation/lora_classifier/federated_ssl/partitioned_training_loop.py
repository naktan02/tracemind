"""Compatibility shim for legacy partitioned training loop imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.federated_ssl.partitioned.training_loop import (
    HelperWeakProbabilityProvider,
    PartitionedLoraStepResult,
    PartitionedLoraTrainingResult,
    run_partitioned_lora_classifier_step,
    run_physical_partitioned_adapter_classifier_step,
    train_partitioned_lora_classifier,
    train_physical_partitioned_adapter_classifier,
)
