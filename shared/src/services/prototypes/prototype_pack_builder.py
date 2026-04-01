"""카테고리별 임베딩에서 single-centroid PrototypePack을 만드는 빌드 유틸리티.

exact incremental build-state를 지원하는 현재 single 전용 builder다.
multi-prototype 생성은 상위 build strategy 계층에서 다룬다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from shared.src.contracts.prototype_build_state_contracts import (
    CategoryPrototypeBuildStatePayload,
    PrototypeBuildStatePayload,
)
from shared.src.domain.entities.artifacts.prototype_pack import (
    CategoryPrototype,
    PrototypePack,
)


@dataclass(slots=True)
class PrototypePackBuilder:
    """카테고리별 임베딩 묶음을 single centroid prototype pack으로 변환한다."""

    pack_schema_version: str = "prototype_pack.v1"
    build_state_schema_version: str = "prototype_build_state.v1"
    build_method: str = "mean_centroid_l2_normalized"
    distance_metric: str = "cosine"

    def build(
        self,
        embeddings_by_category: Mapping[str, Sequence[Sequence[float]]],
        *,
        prototype_version: str,
        embedding_backend: str = "transformers_mxbai",
        embedding_model_id: str,
        embedding_model_revision: str,
        normalize_embeddings: bool = True,
        task_prefix: str = "",
        translation_model_id: str | None,
        translation_model_revision: str | None,
        translation_direction: str | None,
        mapping_version: str,
        built_at: datetime,
        required_categories: Sequence[str] | None = None,
    ) -> PrototypePack:
        build_state = self.build_state(
            embeddings_by_category,
            prototype_version=prototype_version,
            embedding_backend=embedding_backend,
            embedding_model_id=embedding_model_id,
            embedding_model_revision=embedding_model_revision,
            normalize_embeddings=normalize_embeddings,
            task_prefix=task_prefix,
            translation_model_id=translation_model_id,
            translation_model_revision=translation_model_revision,
            translation_direction=translation_direction,
            mapping_version=mapping_version,
            built_at=built_at,
            required_categories=required_categories,
        )
        return self.build_pack_from_state(build_state)

    def build_state(
        self,
        embeddings_by_category: Mapping[str, Sequence[Sequence[float]]],
        *,
        prototype_version: str,
        embedding_backend: str = "transformers_mxbai",
        embedding_model_id: str,
        embedding_model_revision: str,
        normalize_embeddings: bool = True,
        task_prefix: str = "",
        translation_model_id: str | None,
        translation_model_revision: str | None,
        translation_direction: str | None,
        mapping_version: str,
        built_at: datetime,
        required_categories: Sequence[str] | None = None,
    ) -> PrototypeBuildStatePayload:
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
            raise ValueError(
                "At least one category is required to build a prototype pack."
            )

        categories: dict[str, CategoryPrototypeBuildStatePayload] = {}
        for category in categories_to_build:
            bucket = embeddings_by_category.get(category)
            if not bucket:
                raise ValueError(
                    f"Category '{category}' has no embeddings to build from."
                )

            embedding_sum, sample_count = self._sum_embeddings(
                bucket,
                category=category,
            )
            categories[category] = CategoryPrototypeBuildStatePayload(
                embedding_sum=embedding_sum,
                sample_count=sample_count,
            )

        return PrototypeBuildStatePayload(
            schema_version=self.build_state_schema_version,
            prototype_version=prototype_version,
            embedding_backend=embedding_backend,
            embedding_model_id=embedding_model_id,
            embedding_model_revision=embedding_model_revision,
            normalize_embeddings=normalize_embeddings,
            task_prefix=task_prefix,
            translation_model_id=translation_model_id,
            translation_model_revision=translation_model_revision,
            translation_direction=translation_direction,
            mapping_version=mapping_version,
            build_method=self.build_method,
            distance_metric=self.distance_metric,
            built_at=built_at,
            categories=categories,
        )

    def build_pack_from_state(
        self,
        build_state: PrototypeBuildStatePayload,
    ) -> PrototypePack:
        categories: dict[str, CategoryPrototype] = {}
        for category, state in build_state.categories.items():
            centroid = self._normalized_centroid_from_sum(
                state.embedding_sum,
                sample_count=state.sample_count,
                category=category,
            )
            categories[category] = CategoryPrototype(
                centroid=centroid,
                sample_count=state.sample_count,
            )

        return PrototypePack(
            schema_version=self.pack_schema_version,
            prototype_version=build_state.prototype_version,
            embedding_model_id=build_state.embedding_model_id,
            embedding_model_revision=build_state.embedding_model_revision,
            translation_model_id=build_state.translation_model_id,
            translation_model_revision=build_state.translation_model_revision,
            translation_direction=build_state.translation_direction,
            mapping_version=build_state.mapping_version,
            build_method=build_state.build_method,
            distance_metric=build_state.distance_metric,
            built_at=build_state.built_at,
            categories=categories,
        )

    def merge_build_state(
        self,
        base_state: PrototypeBuildStatePayload,
        embeddings_by_category: Mapping[str, Sequence[Sequence[float]]],
        *,
        prototype_version: str,
        built_at: datetime,
        required_categories: Sequence[str] | None = None,
    ) -> PrototypeBuildStatePayload:
        if base_state.build_method != self.build_method:
            raise ValueError(
                "Prototype build_state build_method does not match the current builder."
            )
        if base_state.distance_metric != self.distance_metric:
            raise ValueError(
                "Prototype build_state distance_metric does not match "
                "the current builder."
            )

        categories_to_build = (
            list(required_categories)
            if required_categories is not None
            else sorted(base_state.categories)
        )
        if not categories_to_build:
            raise ValueError(
                "At least one category is required to merge a build state."
            )

        unexpected_categories = sorted(
            category
            for category in embeddings_by_category
            if category not in categories_to_build
        )
        if unexpected_categories:
            raise ValueError(
                f"Unexpected categories for merge: {unexpected_categories}"
            )

        categories: dict[str, CategoryPrototypeBuildStatePayload] = {}
        for category in categories_to_build:
            base_category_state = base_state.categories.get(category)
            if base_category_state is None:
                raise ValueError(
                    f"Category '{category}' does not exist in the base build state."
                )

            total_sum = list(base_category_state.embedding_sum)
            total_count = base_category_state.sample_count
            new_bucket = embeddings_by_category.get(category, ())
            if new_bucket:
                new_sum, new_count = self._sum_embeddings(new_bucket, category=category)
                if len(new_sum) != len(total_sum):
                    raise ValueError(
                        f"Category '{category}' contains embeddings with "
                        "mismatched dimensions."
                    )
                total_sum = [
                    left + right
                    for left, right in zip(total_sum, new_sum, strict=True)
                ]
                total_count += new_count

            categories[category] = CategoryPrototypeBuildStatePayload(
                embedding_sum=total_sum,
                sample_count=total_count,
            )

        return PrototypeBuildStatePayload(
            schema_version=base_state.schema_version,
            prototype_version=prototype_version,
            embedding_backend=base_state.embedding_backend,
            embedding_model_id=base_state.embedding_model_id,
            embedding_model_revision=base_state.embedding_model_revision,
            normalize_embeddings=base_state.normalize_embeddings,
            task_prefix=base_state.task_prefix,
            translation_model_id=base_state.translation_model_id,
            translation_model_revision=base_state.translation_model_revision,
            translation_direction=base_state.translation_direction,
            mapping_version=base_state.mapping_version,
            build_method=base_state.build_method,
            distance_metric=base_state.distance_metric,
            built_at=built_at,
            categories=categories,
        )

    @staticmethod
    def _sum_embeddings(
        embeddings: Sequence[Sequence[float]],
        *,
        category: str,
    ) -> tuple[list[float], int]:
        first_vector = tuple(float(value) for value in embeddings[0])
        if not first_vector:
            raise ValueError(f"Category '{category}' contains an empty embedding.")

        totals = [0.0] * len(first_vector)
        for embedding in embeddings:
            vector = tuple(float(value) for value in embedding)
            if len(vector) != len(first_vector):
                raise ValueError(
                    f"Category '{category}' contains embeddings with "
                    "mismatched dimensions."
                )
            for index, value in enumerate(vector):
                totals[index] += value

        return totals, len(embeddings)

    @staticmethod
    def _normalized_centroid_from_sum(
        embedding_sum: Sequence[float],
        *,
        sample_count: int,
        category: str,
    ) -> list[float]:
        if sample_count <= 0:
            raise ValueError(
                f"Category '{category}' sample_count must be positive, "
                f"got {sample_count}."
            )

        centroid = [value / sample_count for value in embedding_sum]
        norm = math.sqrt(sum(value * value for value in centroid))
        if norm == 0.0:
            raise ValueError(
                f"Category '{category}' produces a zero norm centroid and "
                "cannot be normalized."
            )
        return [value / norm for value in centroid]
