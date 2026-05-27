"""PEFT-backed classifier runtime resource cache keys."""

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
    """PEFT-backed classifier runtime resource cache key를 만든다."""

    payload = json.dumps(dict(values), sort_keys=True, separators=(",", ":"))
    return f"{PEFT_ENCODER_RESOURCE_CACHE_NAMESPACE}:{kind}:{payload}"


def peft_encoder_resource_cache_prefix(kind: str) -> str:
    """canonical PEFT-backed classifier runtime resource prefix."""

    return f"{PEFT_ENCODER_RESOURCE_CACHE_NAMESPACE}:{kind}:"


def clear_peft_encoder_transient_resource_cache(runtime_resource_cache: object) -> int:
    """round 뒤 버려도 되는 무거운 PEFT-backed classifier cache를 비운다."""

    clear_resources = getattr(runtime_resource_cache, "clear_resources", None)
    if not callable(clear_resources):
        return 0
    removed = 0
    for kind in PEFT_ENCODER_TRANSIENT_RESOURCE_KINDS:
        removed += int(
            clear_resources(key_prefix=peft_encoder_resource_cache_prefix(kind))
        )
    return removed
