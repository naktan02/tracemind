"""Federated SSL method descriptor helpers."""

from .registry import (
    FederatedSslMethodDescriptor,
    resolve_federated_ssl_method,
)

__all__ = [
    "FederatedSslMethodDescriptor",
    "resolve_federated_ssl_method",
]
