"""Legacy LoRA-classifier partitioned objective training compatibility shim."""

# ruff: noqa: F401,E501

from methods.adaptation.peft_text_classifier.federated_ssl.partitioned_objective_training import (
    PartitionedLocalRuntimePlan,
    PartitionedMethodLocalTrainingConfig,
    replace_partitioned_training_deltas,
    run_partitioned_lora_classifier_training_core,
)
from methods.federated_ssl.fedmatch.partitioned_local_training import (
    run_method_owned_lora_classifier_training_core,
)
