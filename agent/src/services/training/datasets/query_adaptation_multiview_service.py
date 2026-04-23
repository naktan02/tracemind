"""Single-view query adaptation dataset을 multiview source row로 확장한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from agent.src.services.training.backends.inputs.models import (
    TrainingExampleSource,
)
from agent.src.services.training.datasets.query_adaptation_dataset_service import (
    QueryAdaptationDataset,
    QueryAdaptationDatasetExample,
)

_MetadataScalar = str | int | float | bool


@dataclass(slots=True)
class QueryAdaptationMultiviewViews:
    """한 query example의 weak/strong view 묶음."""

    augmenter_name: str
    weak_text: str
    strong_text: str
    weak_translated_text: str | None = None
    strong_translated_text: str | None = None
    metadata: dict[str, _MetadataScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.augmenter_name.strip():
            raise ValueError("augmenter_name must not be empty.")
        if not self.weak_text.strip():
            raise ValueError("weak_text must not be empty.")
        if not self.strong_text.strip():
            raise ValueError("strong_text must not be empty.")


class QueryAdaptationMultiviewAugmenter(Protocol):
    """Query adaptation example에서 weak/strong view를 생성한다."""

    augmenter_name: str

    def build_views(
        self,
        *,
        example: QueryAdaptationDatasetExample,
    ) -> QueryAdaptationMultiviewViews:
        """단일 query adaptation example을 multiview로 확장한다."""


@dataclass(slots=True)
class IdentityQueryAdaptationMultiviewAugmenter:
    """Weak/strong을 원문 그대로 복사하는 pass-through augmenter."""

    augmenter_name: str = "identity_multiview"

    def build_views(
        self,
        *,
        example: QueryAdaptationDatasetExample,
    ) -> QueryAdaptationMultiviewViews:
        return QueryAdaptationMultiviewViews(
            augmenter_name=self.augmenter_name,
            weak_text=example.source_row.text,
            strong_text=example.source_row.text,
            weak_translated_text=example.source_row.translated_text,
            strong_translated_text=example.source_row.translated_text,
        )


@dataclass(slots=True)
class QueryAdaptationMultiviewExample:
    """Single-view adaptation example에 multiview 입력을 덧붙인 파생 row."""

    base_example: QueryAdaptationDatasetExample
    views: QueryAdaptationMultiviewViews

    @property
    def query_id(self) -> str:
        return self.base_example.query_id

    @property
    def source_row(self) -> TrainingExampleSource:
        return TrainingExampleSource(
            query_id=self.base_example.source_row.query_id,
            text=self.base_example.source_row.text,
            occurred_at=self.base_example.source_row.occurred_at,
            translated_text=self.base_example.source_row.translated_text,
            weak_text=self.views.weak_text,
            strong_text=self.views.strong_text,
            weak_translated_text=self.views.weak_translated_text,
            strong_translated_text=self.views.strong_translated_text,
        )


@dataclass(slots=True)
class QueryAdaptationMultiviewDataset:
    """Multiview training input으로 바로 넘길 수 있는 query adaptation 묶음."""

    examples: tuple[QueryAdaptationMultiviewExample, ...]

    @property
    def count(self) -> int:
        return len(self.examples)

    @property
    def source_rows(self) -> tuple[TrainingExampleSource, ...]:
        return tuple(example.source_row for example in self.examples)


@dataclass(slots=True)
class QueryAdaptationMultiviewService:
    """Single-view adaptation dataset을 multiview row로 확장한다."""

    augmenter: QueryAdaptationMultiviewAugmenter = field(
        default_factory=IdentityQueryAdaptationMultiviewAugmenter
    )

    def build_dataset(
        self,
        *,
        dataset: QueryAdaptationDataset,
    ) -> QueryAdaptationMultiviewDataset:
        examples: list[QueryAdaptationMultiviewExample] = []
        for base_example in dataset.examples:
            views = self.augmenter.build_views(example=base_example)
            if views.augmenter_name != self.augmenter.augmenter_name:
                raise ValueError(
                    "Returned multiview augmenter_name does not match the "
                    "augmenter instance."
                )
            examples.append(
                QueryAdaptationMultiviewExample(
                    base_example=base_example,
                    views=views,
                )
            )
        return QueryAdaptationMultiviewDataset(examples=tuple(examples))


__all__ = [
    "IdentityQueryAdaptationMultiviewAugmenter",
    "QueryAdaptationMultiviewAugmenter",
    "QueryAdaptationMultiviewDataset",
    "QueryAdaptationMultiviewExample",
    "QueryAdaptationMultiviewService",
    "QueryAdaptationMultiviewViews",
]
