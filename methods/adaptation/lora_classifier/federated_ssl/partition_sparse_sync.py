"""Compatibility shim for legacy partition sparse sync imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.federated_ssl.partitioned.sparse_sync import (
    PartitionSparseSyncParameters,
    PartitionSparseUploadProjection,
    apply_partitioned_c2s_sparse_upload,
    apply_partitioned_s2c_sparse_download,
    count_partition_delta_nonzero_values,
    project_partitioned_c2s_sparse_upload,
    project_partitioned_s2c_sparse_download,
)
