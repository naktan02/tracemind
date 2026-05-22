"""Current shared adapter state contract tests."""

from __future__ import annotations

from shared.src.contracts.adapter_contract_families.base import (
    CurrentSharedAdapterStatePayload,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_current_shared_adapter_state_payload,
    make_identity_state_payload,
)
from shared.src.contracts.model_contracts import make_embedding_manifest


def test_current_shared_adapter_state_parses_registered_state_payload() -> None:
    state = make_identity_state_payload(
        model_id="model",
        model_revision="rev_001",
        embedding_dim=2,
    )
    manifest = make_embedding_manifest(
        model_id="model",
        model_revision="rev_001",
        auxiliary_artifact_versions={"prototype_pack": "proto_001"},
        artifact_ref="shared_adapter_state::rev_001",
    )
    payload = make_current_shared_adapter_state_payload(
        manifest=manifest,
        state=state,
    )

    parsed = CurrentSharedAdapterStatePayload.model_validate(
        payload.model_dump(mode="json")
    )

    assert parsed.manifest.model_revision == "rev_001"
    assert parsed.state.adapter_kind == "diagonal_scale"
    assert parsed.state.embedding_dim == 2


def test_current_shared_adapter_state_rejects_revision_mismatch() -> None:
    state = make_identity_state_payload(
        model_id="model",
        model_revision="rev_state",
        embedding_dim=2,
    )
    manifest = make_embedding_manifest(
        model_id="model",
        model_revision="rev_manifest",
        auxiliary_artifact_versions={"prototype_pack": "proto_001"},
        artifact_ref="shared_adapter_state::rev_manifest",
    )

    try:
        make_current_shared_adapter_state_payload(
            manifest=manifest,
            state=state,
        )
    except ValueError as error:
        assert "model_revision" in str(error)
    else:
        raise AssertionError("Expected revision mismatch to fail.")
