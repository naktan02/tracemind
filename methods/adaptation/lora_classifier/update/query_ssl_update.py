"""Compatibility shim for legacy lora_classifier Query SSL update imports."""

# ruff: noqa: F401,E501,I001

from methods.adaptation.text_classifier.peft_encoder.update.query_ssl_update import (
    QuerySslLoraUpdateBuildResult,
    build_query_ssl_lora_client_metrics,
    build_query_ssl_lora_update_payload,
)
