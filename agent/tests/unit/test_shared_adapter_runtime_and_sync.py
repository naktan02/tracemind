"""Shared adapter runtime/sync unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from agent.src.infrastructure.repositories.shared_adapter_state_repository import (
    SharedAdapterStateRepository,
)
from agent.src.services.assets.shared_adapters.runtime_service import (
    SharedAdapterRuntimeService,
)
from agent.src.services.assets.shared_adapters.sync_service import (
    SharedAdapterSyncService,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_current_shared_adapter_state_payload,
    make_peft_classifier_state_payload,
)
from shared.src.contracts.model_contracts import make_embedding_manifest


def _peft_state(*, model_revision: str):
    return make_peft_classifier_state_payload(
        model_id="model",
        model_revision=model_revision,
        backbone={
            "backbone_model_id": "mixedbread-ai/mxbai-embed-large-v1",
            "backbone_revision": "main",
            "tokenizer_model_id": "mixedbread-ai/mxbai-embed-large-v1",
            "tokenizer_revision": "main",
            "pooling": "mean",
            "max_length": 256,
            "task_prefix": "",
        },
        peft_adapter_config={
            "peft_adapter_name": "lora",
            "parameters": {"rank": 8},
        },
        label_schema=["anxiety", "normal"],
    )


def _current_payload() -> dict:
    state = _peft_state(model_revision="rev_001")
    manifest = make_embedding_manifest(
        model_id="model",
        model_revision="rev_001",
        auxiliary_artifact_versions={"calibration_set": "calib_001"},
        artifact_ref="/server/state/rev_001.json",
        training_scope=state.training_scope,
    )
    return make_current_shared_adapter_state_payload(
        manifest=manifest,
        state=state,
    ).model_dump(mode="json")


def test_shared_adapter_runtime_reads_active_state(tmp_path: Path) -> None:
    repository = SharedAdapterStateRepository(state_root=tmp_path / "shared_states")
    current = _current_payload()
    state = _peft_state(model_revision="rev_001")
    repository.save_current(
        manifest=make_embedding_manifest(
            model_id="model",
            model_revision="rev_001",
            auxiliary_artifact_versions={"calibration_set": "calib_001"},
            artifact_ref="/server/state/rev_001.json",
            training_scope=state.training_scope,
        ),
        state=state,
    )

    runtime = SharedAdapterRuntimeService(repository=repository)

    assert (
        runtime.get_active_manifest().model_revision
        == current["manifest"]["model_revision"]
    )
    assert runtime.get_active_state().model_revision == "rev_001"


def test_shared_adapter_sync_pulls_current_state_and_activates_it(
    tmp_path: Path,
) -> None:
    repository = SharedAdapterStateRepository(state_root=tmp_path / "shared_states")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/fl/rounds/active-state/current"
        return httpx.Response(
            status_code=200,
            request=request,
            content=json.dumps(_current_payload()).encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    sync_service = SharedAdapterSyncService(
        repository=repository,
        _transport=httpx.MockTransport(handler),
    )

    pointer = sync_service.pull_current(server_base_url="http://server.test")

    assert pointer.model_revision == "rev_001"
    assert repository.path_for_revision("rev_001").exists()
    assert repository.manifest_path_for_revision("rev_001").exists()
    assert repository.load_active_state().model_revision == "rev_001"
