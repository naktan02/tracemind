"""Agent runtime profile contract tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from shared.src.contracts.agent_runtime_profile_contracts import (
    AgentRuntimeProfilePayload,
    AgentRuntimeProfileValidationResponsePayload,
    make_agent_runtime_profile_payload,
)


def _profile(
    *,
    profile_revision: str = "runtime_rev_001",
    model_revision: str = "clf_2026_04_11_143138",
) -> AgentRuntimeProfilePayload:
    return make_agent_runtime_profile_payload(
        profile_id="profile_peft_classifier_lora",
        profile_revision=profile_revision,
        model_id="mixedbread-ai/mxbai-embed-large-v1",
        model_revision=model_revision,
        runtime_family="peft_classifier",
        adapter_mechanism="lora",
        scorer_backend_name="classifier_head_logits",
        embedding_backend="transformers_mxbai",
        embedding_model_id="mixedbread-ai/mxbai-embed-large-v1",
        training_scope="adapter_and_head",
        required_state_kind="peft_classifier_state.v2",
        updated_at=datetime(2026, 6, 13, tzinfo=timezone.utc),
    )


def test_runtime_profile_checksum_is_payload_identity() -> None:
    profile = _profile()

    loaded = AgentRuntimeProfilePayload.model_validate_json(profile.model_dump_json())

    assert loaded.identity_matches(profile)
    assert loaded.payload_checksum == profile.payload_checksum


def test_runtime_profile_checksum_ignores_updated_at() -> None:
    profile = _profile()
    payload = profile.model_dump(mode="json")
    payload["updated_at"] = datetime(2026, 6, 14, tzinfo=timezone.utc).isoformat()

    loaded = AgentRuntimeProfilePayload.model_validate(payload)

    assert loaded.identity_matches(profile)
    assert loaded.payload_checksum == profile.payload_checksum


def test_runtime_profile_rejects_tampered_checksum() -> None:
    payload = _profile().model_dump(mode="json")
    payload["scorer_backend_name"] = "prototype_similarity"

    with pytest.raises(ValueError, match="payload_checksum"):
        AgentRuntimeProfilePayload.model_validate(payload)


def test_validation_response_requires_latest_profile_when_stale() -> None:
    with pytest.raises(ValueError, match="latest_profile"):
        AgentRuntimeProfileValidationResponsePayload(up_to_date=False)

    response = AgentRuntimeProfileValidationResponsePayload(
        up_to_date=False,
        latest_profile=_profile(profile_revision="runtime_rev_002"),
    )

    assert response.latest_profile is not None
    assert response.latest_profile.profile_revision == "runtime_rev_002"
