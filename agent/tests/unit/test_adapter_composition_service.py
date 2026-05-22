"""Adapter composition service 단위 테스트."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import pytest

from agent.src.services.assets.adapters.composition_service import (
    AdapterCompositionService,
)
from shared.src.contracts.model_contracts import make_embedding_manifest


@dataclass(slots=True)
class _FakeAdapterState:
    model_id: str
    model_revision: str
    adapter_kind: str
    factor: float
    training_scope: str = "adapter_only"
    updated_at: datetime = datetime(2026, 4, 1, tzinfo=timezone.utc)
    schema_version: str = "fake_adapter_state.v1"
    seen_inputs: list[list[float]] = field(default_factory=list)

    @property
    def embedding_dim(self) -> int:
        return 2

    def apply(self, embedding: Sequence[float]) -> list[float]:
        values = [float(value) for value in embedding]
        self.seen_inputs.append(values)
        return [value * self.factor for value in values]


@dataclass(slots=True)
class _SharedProvider:
    state: _FakeAdapterState

    def get_active_state(self) -> _FakeAdapterState:
        return self.state

    def get_active_manifest(self):
        return make_embedding_manifest(
            model_id=self.state.model_id,
            model_revision=self.state.model_revision,
            auxiliary_artifact_versions={"prototype_pack": "proto_001"},
            artifact_ref=f"shared_adapter_state::{self.state.model_revision}",
        )


@dataclass(slots=True)
class _LocalProvider:
    state: _FakeAdapterState

    def get_active_state(self) -> _FakeAdapterState:
        return self.state


def test_adapter_composition_applies_shared_then_local_state() -> None:
    shared_state = _FakeAdapterState(
        model_id="model",
        model_revision="shared_rev",
        adapter_kind="global_fake",
        factor=2.0,
    )
    local_state = _FakeAdapterState(
        model_id="model",
        model_revision="local_rev",
        adapter_kind="local_fake",
        factor=3.0,
    )
    context = AdapterCompositionService(
        shared_adapter_provider=_SharedProvider(shared_state),
        local_adapter_provider=_LocalProvider(local_state),
    ).get_context()

    result = context.apply_for_inference([1.0, 2.0])

    assert result == [6.0, 12.0]
    assert shared_state.seen_inputs == [[1.0, 2.0]]
    assert local_state.seen_inputs == [[2.0, 4.0]]
    assert context.query_buffer_metadata() == {
        "adapter_kind": "global_fake",
        "shared_adapter_kind": "global_fake",
        "shared_model_revision": "shared_rev",
        "local_adapter_kind": "local_fake",
        "local_adapter_revision": "local_rev",
    }
    assert context.model_revision_for_record("fallback") == "shared_rev"


def test_adapter_composition_keeps_local_state_optional() -> None:
    context = AdapterCompositionService().get_context()

    assert context.apply_for_inference([1.0, 2.0]) == [1.0, 2.0]
    assert context.query_buffer_metadata() == {}
    assert context.model_revision_for_record("fallback") == "fallback"


def test_adapter_composition_requires_shared_context_for_training() -> None:
    with pytest.raises(FileNotFoundError, match="shared adapter provider"):
        AdapterCompositionService().get_context(require_shared=True)
