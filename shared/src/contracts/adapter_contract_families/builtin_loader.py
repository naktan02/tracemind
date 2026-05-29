"""Explicit builtin loader for shared adapter payload families."""

from __future__ import annotations

from .base import AdapterKind
from .classifier_head import (
    ClassifierHeadAdapterStatePayload,
    ClassifierHeadAdapterUpdatePayload,
)
from .peft_classifier import (
    PeftClassifierAdapterStatePayload,
    PeftClassifierAdapterUpdatePayload,
)
from .registry import register_shared_adapter_payload_family

_BUILTINS_LOADED = False


def load_builtin_shared_adapter_payload_families() -> None:
    """Register builtin shared adapter payload families exactly once."""

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    register_shared_adapter_payload_family(
        AdapterKind.CLASSIFIER_HEAD.value,
        state_payload_type=ClassifierHeadAdapterStatePayload,
        update_payload_type=ClassifierHeadAdapterUpdatePayload,
    )
    register_shared_adapter_payload_family(
        AdapterKind.PEFT_CLASSIFIER.value,
        state_payload_type=PeftClassifierAdapterStatePayload,
        update_payload_type=PeftClassifierAdapterUpdatePayload,
    )
    _BUILTINS_LOADED = True
