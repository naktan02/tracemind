"""카테고리별 임베딩에서 PrototypePack을 만드는 도메인 서비스."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from src.domain.entities.prototype_pack import CategoryPrototype, PrototypePack


@dataclass(slots=True)
class PrototypePackBuilder:
    """카테고리별 임베딩 묶음을 centroid 기반 prototype pack으로 변환한다."""

    schema_version: str = "prototype_pack.v1"
    build_method: str = "mean_centroid"
    distance_metric: str = "cosine"

    def build(
        self,
        embeddings_by_category: Mapping[str, Sequence[Sequence[float]]],
        *,
        prototype_version: str,
        embedding_model_id: str,
        embedding_model_revision: str,
        translation_model_id: str | None,
        translation_model_revision: str | None,
        translation_direction: str | None,
        mapping_version: str,
        built_at: datetime,
        required_categories: Sequence[str] | None = None,
    ) -> PrototypePack:
        if not prototype_version.strip():
            raise ValueError("prototype_version must not be empty.")
        if not embedding_model_id.strip():
            raise ValueError("embedding_model_id must not be empty.")
        if not mapping_version.strip():
            raise ValueError("mapping_version must not be empty.")

        categories_to_build = (
            list(required_categories)
            if required_categories is not None
            else sorted(embeddings_by_category)
        )
        if not categories_to_build:
            raise ValueError("At least one category is required to build a prototype pack.")

        categories: dict[str, CategoryPrototype] = {}
        for category in categories_to_build:
            bucket = embeddings_by_category.get(category)
            if not bucket:
                raise ValueError(f"Category '{category}' has no embeddings to build from.")

            centroid = self._mean_centroid(bucket, category=category)
            categories[category] = CategoryPrototype(
                centroid=centroid,
                sample_count=len(bucket),
            )

        return PrototypePack(
            schema_version=self.schema_version,
            prototype_version=prototype_version,
            embedding_model_id=embedding_model_id,
            embedding_model_revision=embedding_model_revision,
            translation_model_id=translation_model_id,
            translation_model_revision=translation_model_revision,
            translation_direction=translation_direction,
            mapping_version=mapping_version,
            build_method=self.build_method,
            distance_metric=self.distance_metric,
            built_at=built_at,
            categories=categories,
        )

    @staticmethod
    def _mean_centroid(
        embeddings: Sequence[Sequence[float]],
        *,
        category: str,
    ) -> list[float]:
        first_vector = tuple(float(value) for value in embeddings[0])
        if not first_vector:
            raise ValueError(f"Category '{category}' contains an empty embedding.")

        totals = [0.0] * len(first_vector)
        for embedding in embeddings:
            vector = tuple(float(value) for value in embedding)
            if len(vector) != len(first_vector):
                raise ValueError(
                    f"Category '{category}' contains embeddings with mismatched dimensions."
                )
            for index, value in enumerate(vector):
                totals[index] += value

        count = len(embeddings)
        return [value / count for value in totals]
