"""Compatibility shim for Query SSL LoRA-classifier update imports."""

from __future__ import annotations

from methods.adaptation.lora_classifier.update.query_ssl_update import (
    QuerySslLoraUpdateBuildResult as QuerySslLoraUpdateBuildResult,
)
from methods.adaptation.lora_classifier.update.query_ssl_update import (
    build_query_ssl_lora_client_metrics as build_query_ssl_lora_client_metrics,
)
from methods.adaptation.lora_classifier.update.query_ssl_update import (
    build_query_ssl_lora_update_payload as build_query_ssl_lora_update_payload,
)
