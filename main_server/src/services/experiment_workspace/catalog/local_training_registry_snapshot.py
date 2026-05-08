"""Compatibility catalog snapshot for agent-owned local runtime registries.

`main_server` workspace catalog는 architecture guard상 `agent` 구현을 직접 import할
수 없으므로 이 snapshot을 읽는다. Agent runtime registry wiring의 catalog entry는
각 implementation module 옆에 둔다. 이 파일은 long-term source of truth가 아니라
cross-layer catalog compatibility facade다.

methods-owned local update/scoring backend catalog는 이 파일에 복제하지 않고
methods registry에 위임한다.

새 experiment-only backend나 method-specific runtime을 여기에 추가하지 않는다. 새 항목이
필요하면 먼저 implementation-local catalog entry와 runtime capability 경계를 정하고,
workspace UI/catalog 노출이 필요한 stable 항목만 snapshot으로 반영한다.
"""

from __future__ import annotations

from methods.adaptation import local_update_registry, scoring_registry
from shared.src.contracts.registry_catalog_metadata import (
    RegistryCatalogEntry,
    dedupe_registry_catalog_entries,
)

ANY_ADAPTER_KIND = "*"

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


def list_shared_adapter_training_backend_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """methods-owned local update backend catalog entry 목록."""

    return local_update_registry.list_shared_adapter_training_backend_catalog_entries()


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
    """Agent-local scorer와 methods-owned shared adapter scorer catalog 목록."""

    return dedupe_registry_catalog_entries(
        (
            PROTOTYPE_SIMILARITY_SCORING_BACKEND_CATALOG_ENTRY,
            *scoring_registry.list_shared_adapter_scoring_backend_catalog_entries(),
        )
    )
