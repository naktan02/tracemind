"""Shared adapter contract compatibility facade."""
# ruff: noqa: F401

from __future__ import annotations

from shared.src.contracts.adapter_contract_families.base import (
    CLASSIFIER_HEAD_DELTA_V1,
    CLASSIFIER_HEAD_STATE_V1,
    CURRENT_SHARED_ADAPTER_STATE_V1,
    LORA_CLASSIFIER_DELTA_V1,
    LORA_CLASSIFIER_STATE_V1,
    VECTOR_ADAPTER_DELTA_V1,
    VECTOR_ADAPTER_STATE_V1,
    AdapterKind,
    ClassifierHeadDeltaSchemaVersion,
    ClassifierHeadStateSchemaVersion,
    CurrentSharedAdapterStatePayload,
    CurrentSharedAdapterStateSchemaVersion,
    LoraClassifierDeltaSchemaVersion,
    LoraClassifierStateSchemaVersion,
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
    VectorAdapterDeltaSchemaVersion,
    VectorAdapterStateSchemaVersion,
)
from shared.src.contracts.adapter_contract_families.builtin_loader import (
    load_builtin_shared_adapter_payload_families,
)
from shared.src.contracts.adapter_contract_families.classifier_head import (
    ClassifierHeadAdapterStatePayload,
    ClassifierHeadAdapterUpdatePayload,
    ClassifierHeadDelta,
    ClassifierHeadDeltaPayload,
    ClassifierHeadState,
    ClassifierHeadStatePayload,
)
from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    DiagonalScaleAdapterStatePayload,
    DiagonalScaleAdapterUpdatePayload,
    VectorAdapterDelta,
    VectorAdapterDeltaPayload,
    VectorAdapterState,
    VectorAdapterStatePayload,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_classifier_head_delta_payload,
    make_current_shared_adapter_state_payload,
    make_diagonal_delta_payload,
    make_identity_state_payload,
    make_lora_classifier_delta_payload,
    make_lora_classifier_state_payload,
    make_zero_classifier_head_state_payload,
)
from shared.src.contracts.adapter_contract_families.io import (
    dump_shared_adapter_state_payload,
    dump_shared_adapter_update_payload,
    dump_vector_adapter_delta_payload,
    dump_vector_adapter_state_payload,
    load_shared_adapter_state_payload,
    load_shared_adapter_update_payload,
    load_vector_adapter_delta_payload,
    load_vector_adapter_state_payload,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierAdapterStatePayload,
    LoraClassifierAdapterUpdatePayload,
    LoraClassifierBackbonePayload,
    LoraClassifierConfigPayload,
    LoraClassifierDelta,
    LoraClassifierDeltaPayload,
    LoraClassifierState,
    LoraClassifierStatePayload,
)
from shared.src.contracts.adapter_contract_families.registry import (
    get_shared_adapter_canonical_update_payload_format,
    get_shared_adapter_state_payload_type,
    get_shared_adapter_update_payload_formats,
    get_shared_adapter_update_payload_type,
    list_registered_shared_adapter_payload_adapter_kinds,
    parse_shared_adapter_state_payload,
    parse_shared_adapter_update_payload,
    register_shared_adapter_payload_family,
    register_shared_adapter_state_payload_type,
    register_shared_adapter_update_payload_type,
)

load_builtin_shared_adapter_payload_families()
