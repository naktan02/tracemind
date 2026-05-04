"""Prototype rebuild와 publication 조정 서비스."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace

from methods.prototype.building.base import (
    PrototypeBuildRequest,
    PrototypeBuildStrategy,
)
from methods.prototype.building.single import (
    SinglePrototypeBuildStrategy,
)
from shared.src.domain.services.clock import Clock, SystemUtcClock

from .models import (
    PrototypeRebuildInputRecord as PrototypeRebuildInputRecord,
)
from .models import (
    PrototypeRebuildResult,
    ReferencePrototypeRebuildRequest,
)
from .models import (
    ReferencePrototypeSourceRow as ReferencePrototypeSourceRow,
)
from .models import (
    StoredReferencePrototypeRebuildRequest as StoredReferencePrototypeRebuildRequest,
)
from .publication_strategies import (
    InMemoryPrototypePublicationStrategy as InMemoryPrototypePublicationStrategy,
)
from .publication_strategies import (
    PrototypePublicationStrategy,
    ReferenceRebuildPrototypePublicationStrategy,
)
from .stored_input_rebuild_service import (
    PrototypeRebuildInputRepositoryProtocol as PrototypeRebuildInputRepositoryProtocol,
)
from .stored_input_rebuild_service import (
    StoredReferencePrototypeRebuildService as StoredReferencePrototypeRebuildService,
)


@dataclass(slots=True)
class PrototypeRebuildService:
    """runtime builder 선택과 publication 전략을 한곳에서 소유한다."""

    build_strategy: PrototypeBuildStrategy = field(
        default_factory=SinglePrototypeBuildStrategy
    )
    publication_strategy: PrototypePublicationStrategy = field(
        default_factory=ReferenceRebuildPrototypePublicationStrategy
    )
    clock: Clock = field(default_factory=SystemUtcClock)

    def rebuild(self, build_request: PrototypeBuildRequest) -> PrototypeRebuildResult:
        if build_request.built_at is None:
            build_request = replace(build_request, built_at=self.clock.now())

        build_artifacts = self.build_strategy.build(build_request)
        return self.publication_strategy.publish(build_artifacts)

    def rebuild_from_reference_rows(
        self,
        request: ReferencePrototypeRebuildRequest,
    ) -> PrototypeRebuildResult:
        if not request.rows:
            raise ValueError("Reference rebuild rows must not be empty.")

        texts = [row.text for row in request.rows]
        base_embeddings = request.adapter.embed_texts(texts)
        embeddings_by_category: dict[str, list[list[float]]] = defaultdict(list)
        for row, base_embedding in zip(request.rows, base_embeddings, strict=True):
            embeddings_by_category[row.category].append(
                request.adapter_state.apply(base_embedding)
            )

        required_categories = (
            tuple(request.required_categories)
            if request.required_categories is not None
            else tuple(sorted(embeddings_by_category))
        )
        return self.rebuild(
            PrototypeBuildRequest(
                embeddings_by_category=embeddings_by_category,
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
                required_categories=required_categories,
            )
        )
