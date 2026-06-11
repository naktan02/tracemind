"""PEFT text encoder/head runtime resource cache keys."""

from __future__ import annotations

import json
from collections.abc import Mapping

PEFT_ENCODER_RESOURCE_CACHE_NAMESPACE = "peft_encoder"

PEFT_ENCODER_TRANSIENT_RESOURCE_KINDS = (
    "helper_model",
    "backbone_base",
)


def peft_encoder_resource_cache_key(
    *,
    kind: str,
    values: Mapping[str, object],
) -> str:
    """PEFT text encoder/head runtime resource cache key를 만든다."""

    payload = json.dumps(dict(values), sort_keys=True, separators=(",", ":"))
    return f"{PEFT_ENCODER_RESOURCE_CACHE_NAMESPACE}:{kind}:{payload}"


def peft_encoder_resource_cache_prefix(kind: str) -> str:
    """canonical PEFT text encoder/head runtime resource prefix."""

    return f"{PEFT_ENCODER_RESOURCE_CACHE_NAMESPACE}:{kind}:"


def clear_peft_encoder_helper_model_cache(runtime_resource_cache: object) -> int:
    """client 경계에서 FedMatch helper model cache만 비운다."""

    clear_resources = getattr(runtime_resource_cache, "clear_resources", None)
    if not callable(clear_resources):
        return 0
    return int(
        clear_resources(
            key_prefix=peft_encoder_resource_cache_prefix("helper_model")
        )
    )


def clear_peft_encoder_transient_resource_cache(runtime_resource_cache: object) -> int:
    """round 뒤 버려도 되는 무거운 PEFT text encoder/head cache를 비운다."""

    clear_resources = getattr(runtime_resource_cache, "clear_resources", None)
    if not callable(clear_resources):
        return 0
    removed = 0
    for kind in PEFT_ENCODER_TRANSIENT_RESOURCE_KINDS:
        removed += int(
            clear_resources(key_prefix=peft_encoder_resource_cache_prefix(kind))
        )
    return removed
