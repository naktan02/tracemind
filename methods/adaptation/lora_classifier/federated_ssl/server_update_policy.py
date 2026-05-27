"""Compatibility shim for legacy LoRA-classifier server update policy imports."""

# ruff: noqa: F401,E501

from methods.adaptation.peft_text_classifier.federated_ssl.server_update_policy import (
    resolve_lora_classifier_federated_ssl_server_update_backend,
)
