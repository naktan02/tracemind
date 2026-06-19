"""Initial shared-state auxiliary tensor artifact publication service."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor

from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregationArtifactStore,
)

DEFAULT_INITIAL_STATE_ARTIFACT_REF_PREFIX = "server-aggregate://initial-state"


@dataclass(frozen=True, slots=True)
class ServerTensorArtifactSlot:
    """server-owned ref로 저장할 초기 state 보조 tensor artifact."""

    artifact_name: str
    tensors: Mapping[str, Tensor]
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if not str(self.artifact_name).strip():
            raise ValueError("Initial state tensor artifact_name must not be empty.")
        if not self.tensors:
            raise ValueError(
                "Initial state tensor artifact requires at least one tensor."
            )


@dataclass(frozen=True, slots=True)
class InitialStateArtifactPublicationRequest:
    """초기 shared state가 참조할 server-owned tensor artifact 발행 요청."""

    publication_id: str
    artifact_slots: tuple[ServerTensorArtifactSlot, ...]
    artifact_ref_prefix: str = DEFAULT_INITIAL_STATE_ARTIFACT_REF_PREFIX

    def __post_init__(self) -> None:
        if not str(self.publication_id).strip():
            raise ValueError("Initial state artifact publication_id must not be empty.")
        if not self.artifact_slots:
            raise ValueError("Initial state artifact publication requires slots.")
        if not str(self.artifact_ref_prefix).strip():
            raise ValueError(
                "Initial state artifact publication artifact_ref_prefix must not be "
                "empty."
            )


@dataclass(frozen=True, slots=True)
class InitialStateArtifactPublication:
    """초기 state 보조 artifact 발행 결과."""

    publication_id: str
    artifact_refs: dict[str, str]


@dataclass(slots=True)
class InitialStateArtifactPublicationService:
    """초기 shared state용 server-owned tensor artifact 저장을 맡는다."""

    artifact_store: AggregationArtifactStore

    def publish_tensor_artifacts(
        self,
        request: InitialStateArtifactPublicationRequest,
    ) -> InitialStateArtifactPublication:
        """artifact slot별 ref를 만들고 safetensors payload로 저장한다."""

        publication_ref_id = _slug_ref_part(request.publication_id)
        artifact_refs: dict[str, str] = {}
        for slot in request.artifact_slots:
            artifact_name = _slug_ref_part(slot.artifact_name)
            artifact_ref = (
                f"{request.artifact_ref_prefix.rstrip('/')}/"
                f"{publication_ref_id}/{artifact_name}"
            )
            self.artifact_store.save_safetensors_artifact_ref(
                artifact_ref=artifact_ref,
                tensors=dict(slot.tensors),
                metadata=dict(slot.metadata),
            )
            artifact_refs[slot.artifact_name] = artifact_ref
        return InitialStateArtifactPublication(
            publication_id=request.publication_id,
            artifact_refs=artifact_refs,
        )


def _slug_ref_part(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "artifact"
