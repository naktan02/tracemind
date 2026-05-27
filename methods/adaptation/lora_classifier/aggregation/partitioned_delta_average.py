"""Legacy LoRA-classifier partitioned projection import path."""

# ruff: noqa: F401,E501

from methods.adaptation.peft_text_classifier.aggregation.peft_encoder_partitioned_projection import (
    PARTITIONED_DELTA_AVERAGE_BACKEND_NAME,
    LoraClassifierPartitionedDeltaAverageUpdate,
    aggregate_lora_classifier_partitioned_delta_average,
    compute_lora_classifier_partitioned_delta_average,
)
