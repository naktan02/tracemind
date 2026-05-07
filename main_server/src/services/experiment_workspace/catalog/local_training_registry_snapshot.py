"""Compatibility catalog snapshot for agent-owned local training registries.

`main_server` workspace catalog는 architecture guard상 `agent` 구현을 직접 import할
수 없으므로 이 snapshot을 읽는다. Agent runtime registry wiring의 catalog entry는
각 implementation module 옆에 둔다. 이 파일은 long-term source of truth가 아니라
cross-layer catalog compatibility facade다.

새 experiment-only backend나 method-specific runtime을 여기에 추가하지 않는다. 새 항목이
필요하면 먼저 implementation-local catalog entry와 runtime capability 경계를 정하고,
workspace UI/catalog 노출이 필요한 stable 항목만 snapshot으로 반영한다.
"""

from __future__ import annotations

from shared.src.contracts.adapter_family_metadata import (
    CLASSIFIER_HEAD_FAMILY_METADATA,
    DIAGONAL_SCALE_FAMILY_METADATA,
    LORA_CLASSIFIER_FAMILY_METADATA,
)
from shared.src.contracts.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)

ANY_ADAPTER_KIND = "*"

DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="diagonal_scale_heuristic",
    display_name="diagonal_scale_heuristic",
    implementation_module=(
        "agent.src.services.training.backends.training.diagonal_scale_heuristic"
    ),
    core_method_name="diagonal_scale_heuristic",
    family_name=DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,
    supported_adapter_kinds=(DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,),
    accepted_payload_formats=(
        DIAGONAL_SCALE_FAMILY_METADATA.canonical_update_payload_format,
    ),
    metadata={
        "payload_format": (
            DIAGONAL_SCALE_FAMILY_METADATA.canonical_update_payload_format
        )
    },
)

LORA_CLASSIFIER_TRAINING_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="lora_classifier_trainer",
    display_name="lora_classifier_trainer",
    implementation_module=(
        "agent.src.services.training.backends.training.lora_classifier_trainer"
    ),
    core_method_name="lora_classifier_trainer",
    family_name=LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind,
    supported_adapter_kinds=(LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind,),
    accepted_payload_formats=(
        LORA_CLASSIFIER_FAMILY_METADATA.canonical_update_payload_format,
    ),
    tags=("requires_raw_text", "artifact_ref_update"),
    metadata={
        "payload_format": (
            LORA_CLASSIFIER_FAMILY_METADATA.canonical_update_payload_format
        ),
        "requires_raw_text": True,
        "produces_artifact_refs": True,
        "supports_live_stored_event_runtime": False,
    },
)

PROTOTYPE_RESCORE_EXAMPLE_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="prototype_rescore",
    display_name="prototype_rescore",
    implementation_module="agent.src.services.training.backends.inputs.prototype_rescore",
    core_method_name="prototype_rescore",
    family_name="example_generation",
    supported_adapter_kinds=(ANY_ADAPTER_KIND,),
    tags=("supports_stored_event_rebuild",),
    metadata={"supports_stored_event_rebuild": True},
)
WEAK_STRONG_PAIR_EXAMPLE_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="weak_strong_pair",
    display_name="weak_strong_pair",
    implementation_module="agent.src.services.training.backends.inputs.weak_strong_pair",
    core_method_name="weak_strong_pair",
    family_name="example_generation",
    supported_adapter_kinds=(ANY_ADAPTER_KIND,),
    metadata={"supports_stored_event_rebuild": False},
)

PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="prototype_similarity_evidence",
    display_name="prototype_similarity_evidence",
    implementation_module=(
        "agent.src.services.training.backends.evidence.prototype_similarity"
    ),
    core_method_name="prototype_similarity_evidence",
    family_name="pseudo_label_evidence",
    supported_adapter_kinds=(ANY_ADAPTER_KIND,),
)

PROTOTYPE_SIMILARITY_SCORING_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="prototype_similarity",
    display_name="prototype_similarity",
    implementation_module=(
        "agent.src.services.inference.scoring_backends.prototype_similarity"
    ),
    core_method_name="prototype_similarity",
    family_name="scoring",
    supported_adapter_kinds=(ANY_ADAPTER_KIND,),
    metadata={
        "requires_shared_state": False,
        "confidence_kind": "prototype_similarity_top1",
    },
)
CLASSIFIER_HEAD_LOGITS_SCORING_BACKEND_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="classifier_head_logits",
    display_name="classifier_head_logits",
    implementation_module=(
        "agent.src.services.inference.scoring_backends.classifier_head_logits"
    ),
    core_method_name="classifier_head_logits",
    family_name="scoring",
    supported_adapter_kinds=(CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,),
    tags=("requires_shared_state",),
    metadata={
        "requires_shared_state": True,
        "confidence_kind": "classifier_head_logit_top1",
    },
)

TOP1_MARGIN_THRESHOLD_ACCEPTANCE_POLICY_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="top1_margin_threshold",
    display_name="top1_margin_threshold",
    implementation_module="agent.src.services.training.acceptance_policies.top1",
    core_method_name="top1_margin_threshold",
    family_name="pseudo_label_acceptance",
    supported_adapter_kinds=(ANY_ADAPTER_KIND,),
)
TOP1_CONFIDENCE_ONLY_ACCEPTANCE_POLICY_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="top1_confidence_only",
    display_name="top1_confidence_only",
    implementation_module="agent.src.services.training.acceptance_policies.top1",
    core_method_name="top1_confidence_only",
    family_name="pseudo_label_acceptance",
    supported_adapter_kinds=(ANY_ADAPTER_KIND,),
)

DIAGONAL_SCALE_CLIP_ONLY_PRIVACY_GUARD_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="diagonal_scale_clip_only",
    display_name="diagonal_scale_clip_only",
    implementation_module=(
        "agent.src.services.training.execution.privacy_guards."
        "diagonal_scale_clip_only"
    ),
    core_method_name="diagonal_scale_clip_only",
    family_name="privacy_guard",
    supported_adapter_kinds=(DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,),
)
CLASSIFIER_HEAD_CLIP_ONLY_PRIVACY_GUARD_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="classifier_head_clip_only",
    display_name="classifier_head_clip_only",
    implementation_module=(
        "agent.src.services.training.execution.privacy_guards."
        "classifier_head_clip_only"
    ),
    core_method_name="classifier_head_clip_only",
    family_name="privacy_guard",
    supported_adapter_kinds=(CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,),
)
NOOP_PRIVACY_GUARD_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="noop",
    display_name="noop",
    implementation_module=(
        "agent.src.services.training.execution.privacy_guards.noop"
    ),
    core_method_name="noop",
    family_name="privacy_guard",
    supported_adapter_kinds=(ANY_ADAPTER_KIND,),
)


def list_shared_adapter_training_backend_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """Agent local training backend catalog entry 목록."""

    return dedupe_registry_catalog_entries(
        (
            DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CATALOG_ENTRY,
            LORA_CLASSIFIER_TRAINING_BACKEND_CATALOG_ENTRY,
        )
    )


def list_training_example_backend_catalog_entries() -> tuple[RegistryCatalogEntry, ...]:
    """Agent training example backend catalog entry 목록."""

    return dedupe_registry_catalog_entries(
        (
            PROTOTYPE_RESCORE_EXAMPLE_BACKEND_CATALOG_ENTRY,
            WEAK_STRONG_PAIR_EXAMPLE_BACKEND_CATALOG_ENTRY,
        )
    )


def list_pseudo_label_evidence_backend_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """Agent pseudo-label evidence backend catalog entry 목록."""

    return (PROTOTYPE_SIMILARITY_EVIDENCE_BACKEND_CATALOG_ENTRY,)


def list_scoring_backend_catalog_entries() -> tuple[RegistryCatalogEntry, ...]:
    """Agent scoring backend catalog entry 목록."""

    return dedupe_registry_catalog_entries(
        (
            PROTOTYPE_SIMILARITY_SCORING_BACKEND_CATALOG_ENTRY,
            CLASSIFIER_HEAD_LOGITS_SCORING_BACKEND_CATALOG_ENTRY,
        )
    )


def list_pseudo_label_acceptance_policy_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """Agent pseudo-label acceptance policy catalog entry 목록."""

    return dedupe_registry_catalog_entries(
        (
            TOP1_MARGIN_THRESHOLD_ACCEPTANCE_POLICY_CATALOG_ENTRY,
            TOP1_CONFIDENCE_ONLY_ACCEPTANCE_POLICY_CATALOG_ENTRY,
        )
    )


def list_shared_adapter_privacy_guard_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """Agent privacy guard catalog entry 목록."""

    return dedupe_registry_catalog_entries(
        (
            DIAGONAL_SCALE_CLIP_ONLY_PRIVACY_GUARD_CATALOG_ENTRY,
            CLASSIFIER_HEAD_CLIP_ONLY_PRIVACY_GUARD_CATALOG_ENTRY,
            NOOP_PRIVACY_GUARD_CATALOG_ENTRY,
        )
    )
