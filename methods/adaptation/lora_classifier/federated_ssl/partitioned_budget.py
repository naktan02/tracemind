"""Compatibility shim for legacy partitioned budget imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.peft_text_classifier.federated_ssl.partitioned.budget import (
    normalize_partitioned_local_budget_policy,
    resolve_partitioned_local_budget,
)
