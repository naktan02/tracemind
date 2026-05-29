"""Single-centroid prototype build strategy."""

from __future__ import annotations

from dataclasses import dataclass

from methods.prototype.building.base import (
    PrototypeBuildArtifacts,
    PrototypeBuildRequest,
)
from methods.prototype.building.pack_builder import PrototypePackBuilder
from shared.src.services.prototypes.payload_serialization import (
    build_single_prototype_pack_payload,
)


@dataclass(slots=True)
class SinglePrototypeBuildStrategy:
    """기존 exact mean-centroid builder를 감싼 single 전략."""

    name: str = "single"
    supports_exact_build_state: bool = True

    def build(self, request: PrototypeBuildRequest) -> PrototypeBuildArtifacts:
        if request.built_at is None:
            raise ValueError("built_at must not be None.")

        builder = PrototypePackBuilder()
        build_state = builder.build_state(
            request.embeddings_by_category,
            prototype_version=request.prototype_version,
            embedding_backend=request.embedding_backend,
            embedding_model_id=request.embedding_model_id,
            embedding_model_revision=request.embedding_model_revision,
            normalize_embeddings=request.normalize_embeddings,
            task_prefix=request.task_prefix,
            translation_model_id=request.translation_model_id,
            translation_model_revision=request.translation_model_revision,
            translation_direction=request.translation_direction,
            mapping_version=request.mapping_version,
            built_at=request.built_at,
            required_categories=request.required_categories,
        )
        pack = builder.build_pack_from_state(build_state)
        return PrototypeBuildArtifacts(
            pack_payload=build_single_prototype_pack_payload(pack),
            build_state_payload=build_state,
        )
