"""Compatibility shim for legacy partitioned trainable model imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_text_classifier.federated_ssl.partitioned.trainable_model import (
    PARTITION_COMPOSITION_SUM_LOGITS,
    AdapterClassifierPartition,
    AdapterClassifierPartitionSpec,
    PartitionedTrainableAdapterClassifier,
    PartitionedTrainableTextClassifier,
    PartitionedTrainableTextClassifierModules,
    TextClassifierPartitionSpec,
    TextFeatureExtractor,
    TrainableAdapterPartitionPlan,
    parameters_changed,
    snapshot_partition_parameter_tensors,
    snapshot_partition_parameters,
)
