"""Agent-owned local training registry catalog metadata.

이 모듈은 registry 구현체를 import하지 않고 catalog에 필요한 정적 metadata만
소유한다. `main_server` catalog는 이 metadata를 읽고, `agent` registry는 같은
entry를 wiring에 사용해 source-of-truth 중복을 줄인다.
"""

from __future__ import annotations

from shared.src.config.adapter_family_metadata import (
    CLASSIFIER_HEAD_FAMILY_METADATA,
    DIAGONAL_SCALE_FAMILY_METADATA,
    LORA_CLASSIFIER_FAMILY_METADATA,
)
from shared.src.config.registry_catalog_metadata import (
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
    implementation_module="agent.src.services.inference.scoring_backends",
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
    implementation_module="agent.src.services.inference.scoring_backends",
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
    implementation_module="agent.src.services.training.execution.privacy_guard_service",
    core_method_name="diagonal_scale_clip_only",
    family_name="privacy_guard",
    supported_adapter_kinds=(DIAGONAL_SCALE_FAMILY_METADATA.adapter_kind,),
)
CLASSIFIER_HEAD_CLIP_ONLY_PRIVACY_GUARD_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="classifier_head_clip_only",
    display_name="classifier_head_clip_only",
    implementation_module="agent.src.services.training.execution.privacy_guard_service",
    core_method_name="classifier_head_clip_only",
    family_name="privacy_guard",
    supported_adapter_kinds=(CLASSIFIER_HEAD_FAMILY_METADATA.adapter_kind,),
)
NOOP_PRIVACY_GUARD_CATALOG_ENTRY = RegistryCatalogEntry(
    item_name="noop",
    display_name="noop",
    implementation_module="agent.src.services.training.execution.privacy_guard_service",
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
