"""RuntimeProfileRepository tests."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from agent.src.features.runtime_profile.repository import RuntimeProfileRepository
from shared.src.contracts.agent_runtime_profile_contracts import (
    AgentRuntimeProfilePayload,
    make_agent_runtime_profile_payload,
)


def _profile(
    *,
    profile_id: str = "profile_peft_classifier_lora",
    profile_revision: str = "runtime_rev_001",
    scorer_backend_name: str = "peft_classifier_head_logits",
) -> AgentRuntimeProfilePayload:
    return make_agent_runtime_profile_payload(
        profile_id=profile_id,
        profile_revision=profile_revision,
        model_id="mixedbread-ai/mxbai-embed-large-v1",
        model_revision="clf_2026_04_11_143138",
        runtime_family="peft_classifier",
        adapter_mechanism="lora",
        scorer_backend_name=scorer_backend_name,
        embedding_backend="transformers_mxbai",
        embedding_model_id="mixedbread-ai/mxbai-embed-large-v1",
        training_scope="adapter_and_head",
        required_state_kind="peft_classifier_state.v2",
        updated_at=datetime(2026, 6, 13, tzinfo=timezone.utc),
    )


def test_runtime_profile_repository_saves_active_profile(tmp_path: Path) -> None:
    repository = RuntimeProfileRepository(db_path=tmp_path / "agent_local.db")
    received_at = datetime(2026, 6, 13, 1, 0, tzinfo=timezone.utc)

    repository.save_profile(
        _profile(),
        source="server",
        activate=True,
        received_at=received_at,
        server_base_url="http://server.test/",
    )

    active = repository.load_active()
    assert active is not None
    assert active.profile.profile_revision == "runtime_rev_001"
    assert active.source == "server"
    assert active.received_at == received_at
    assert active.activated_at == received_at
    assert active.server_base_url == "http://server.test"


def test_runtime_profile_repository_keeps_single_active_profile(
    tmp_path: Path,
) -> None:
    repository = RuntimeProfileRepository(db_path=tmp_path / "agent_local.db")
    repository.save_profile(
        _profile(profile_revision="rev_1"),
        source="server",
        activate=True,
    )
    repository.save_profile(
        _profile(profile_revision="rev_2", scorer_backend_name="prototype_similarity"),
        source="server",
        activate=True,
    )

    active = repository.load_active()

    assert active is not None
    assert active.profile.profile_revision == "rev_2"
    assert active.profile.scorer_backend_name == "prototype_similarity"


def test_runtime_profile_repository_updates_server_validation_time(
    tmp_path: Path,
) -> None:
    repository = RuntimeProfileRepository(db_path=tmp_path / "agent_local.db")
    profile = _profile()
    validated_at = datetime(2026, 6, 13, 2, 0, tzinfo=timezone.utc)
    repository.save_profile(profile, source="server", activate=True)

    record = repository.mark_server_validated(
        profile_id=profile.profile_id,
        profile_revision=profile.profile_revision,
        payload_checksum=profile.payload_checksum,
        validated_at=validated_at,
        server_base_url="http://server.test",
    )

    assert record.server_validated_at == validated_at
    assert record.server_base_url == "http://server.test"


def test_runtime_profile_repository_adds_server_base_url_to_existing_db(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "agent_local.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE agent_runtime_profiles (
                profile_id          TEXT NOT NULL,
                profile_revision    TEXT NOT NULL,
                payload_checksum    TEXT NOT NULL,
                source              TEXT NOT NULL,
                model_id            TEXT NOT NULL,
                model_revision      TEXT NOT NULL,
                runtime_family      TEXT NOT NULL,
                adapter_mechanism   TEXT,
                scorer_backend_name TEXT NOT NULL,
                embedding_backend   TEXT NOT NULL,
                embedding_model_id  TEXT NOT NULL,
                training_scope      TEXT NOT NULL,
                required_state_kind TEXT,
                payload_json        TEXT NOT NULL,
                received_at         TEXT NOT NULL,
                activated_at        TEXT,
                server_validated_at TEXT,
                is_active           INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (profile_id, profile_revision, payload_checksum)
            );
            """
        )

    repository = RuntimeProfileRepository(db_path=db_path)
    repository.save_profile(
        _profile(),
        source="server",
        activate=True,
        server_base_url="http://server.test",
    )

    active = repository.load_active()
    assert active is not None
    assert active.server_base_url == "http://server.test"
