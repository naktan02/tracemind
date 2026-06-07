"""로컬 inference 파이프라인 서비스.

QueryEvent를 받아 preprocess → [translate] → embed → score → save 흐름을 조합한다.
각 단계는 Protocol로 교체 가능하다.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, Sequence

from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from agent.src.infrastructure.repositories.query_buffer_repository import (
    QueryBufferRecord,
    QueryBufferRepository,
    build_query_buffer_record,
)
from agent.src.services.assets.adapters.composition_service import (
    AdapterCompositionService,
    AdapterRuntimeContext,
    LocalAdapterRuntimeProvider,
    SharedAdapterRuntimeProvider,
)
from agent.src.services.inference.embedding_service import EmbeddingService
from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.language.preprocess_service import PreprocessService
from agent.src.services.language.translation_service import TranslationService
from shared.src.domain.entities.inference.events import AnalysisEvent, QueryEvent


class ScoringAssetProvider(Protocol):
    """scorer backend에 필요한 local asset 조회 인터페이스."""

    def get_scoring_assets(
        self,
    ) -> Mapping[str, Sequence[float] | Sequence[Sequence[float]]]:
        """scorer family별 asset mapping을 반환한다."""
        ...


@dataclass(slots=True)
class InferencePipelineResult:
    """단일 이벤트 파이프라인 실행 결과."""

    analysis_event: AnalysisEvent
    base_embedding: list[float]
    was_translated: bool
    query_buffer_record: QueryBufferRecord | None = None


@dataclass(slots=True)
class InferencePipelineService:
    """QueryEvent → AnalysisEvent 전체 파이프라인을 조합한다.

    각 단계는 독립적으로 교체 가능하다.
    번역은 locale이 translation_locales에 포함될 때만 실행한다.
    """

    embedding_service: EmbeddingService
    scoring_service: ScoringService
    event_repository: AnalysisEventRepository
    scoring_asset_provider: ScoringAssetProvider | None = None
    adapter_composition_service: AdapterCompositionService | None = None
    shared_adapter_provider: SharedAdapterRuntimeProvider | None = None
    local_adapter_provider: LocalAdapterRuntimeProvider | None = None
    query_buffer_repository: QueryBufferRepository | None = None
    preprocess_service: PreprocessService = field(default_factory=PreprocessService)
    translation_service: TranslationService | None = None
    # 번역 대상 locale. 이 목록에 없으면 원문을 그대로 임베딩한다.
    translation_locales: frozenset[str] = frozenset({"ko", "ja", "zh"})
    embedding_model_id: str = "unknown"
    model_revision: str = "unknown"

    def process(self, event: QueryEvent) -> InferencePipelineResult:
        """단일 QueryEvent를 처리하고 결과를 저장한다."""
        normalized = self.preprocess_service.normalize(event.text)

        needs_translation = (
            self.translation_service is not None
            and event.locale in self.translation_locales
        )
        if needs_translation:
            translated_texts = self.translation_service.translate_batch([normalized])  # type: ignore[union-attr]
            text_for_embedding = translated_texts[0]
            translated_text = text_for_embedding
            translation_model_id = _get_translation_model_id(self.translation_service)
        else:
            text_for_embedding = normalized
            translated_text = None
            translation_model_id = None

        embeddings = self.embedding_service.embed_batch([text_for_embedding])
        base_embedding = embeddings[0]

        scoring_assets = self._load_scoring_assets()
        adapter_context = self._load_adapter_context()
        scoring_embedding = adapter_context.apply_for_inference(base_embedding)
        category_scores = self.scoring_service.score(
            scoring_embedding,
            scoring_assets,
            shared_state=adapter_context.shared_state,
        )
        scorer_name = self.scoring_service.backend_name
        confidence_kind = self.scoring_service.confidence_kind
        model_revision = adapter_context.model_revision_for_record(self.model_revision)

        analysis_event = AnalysisEvent(
            query_id=event.query_id,
            occurred_at=event.occurred_at,
            translated_text=translated_text,
            embedding_model_id=self.embedding_model_id,
            translation_model_id=translation_model_id,
            category_scores=category_scores,
        )
        metadata = {
            "embedding_model_id": self.embedding_model_id,
            "translation_model_id": translation_model_id,
            "scorer_backend_name": scorer_name,
            "was_translated": needs_translation,
            "source_locale": event.locale,
            "source_type": event.source_type,
        }
        metadata.update(adapter_context.query_buffer_metadata())
        self.event_repository.save(
            analysis_event,
            base_embedding=list(base_embedding),
            source_event_id=event.query_id,
            scorer_family=_resolve_scorer_family(scorer_name),
            scorer_name=scorer_name,
            model_revision=model_revision,
            confidence_kind=confidence_kind,
            metadata=metadata,
        )
        query_buffer_record = None
        if self.query_buffer_repository is not None:
            query_buffer_record = build_query_buffer_record(
                event=event,
                analysis_event=analysis_event,
                model_revision=model_revision,
                confidence_kind=confidence_kind,
                metadata=metadata,
            )
            self.query_buffer_repository.save(query_buffer_record)
        return InferencePipelineResult(
            analysis_event=analysis_event,
            base_embedding=list(base_embedding),
            was_translated=needs_translation,
            query_buffer_record=query_buffer_record,
        )

    def process_batch(
        self, events: Sequence[QueryEvent]
    ) -> list[InferencePipelineResult]:
        """여러 QueryEvent를 순서대로 처리한다."""
        return [self.process(event) for event in events]

    def _load_adapter_context(self) -> AdapterRuntimeContext:
        if self.adapter_composition_service is not None:
            return self.adapter_composition_service.get_context()
        return AdapterCompositionService(
            shared_adapter_provider=self.shared_adapter_provider,
            local_adapter_provider=self.local_adapter_provider,
        ).get_context()

    def _load_scoring_assets(
        self,
    ) -> Mapping[str, Sequence[float] | Sequence[Sequence[float]]]:
        if self.scoring_asset_provider is not None:
            return self.scoring_asset_provider.get_scoring_assets()
        return {}


def _get_translation_model_id(service: TranslationService) -> str | None:
    """TranslationService 어댑터에서 model_id를 읽는다. 없으면 None."""
    adapter = getattr(service, "adapter", None)
    if adapter is None:
        return None
    model_id = getattr(adapter, "model_id", None)
    return model_id if isinstance(model_id, str) else None


def _resolve_scorer_family(scorer_name: str) -> str:
    """backend name에서 큰 scorer family를 추론한다."""
    lowered = scorer_name.lower()
    if "prototype" in lowered:
        return "prototype"
    if "classifier" in lowered or "linear_head" in lowered:
        return "classifier"
    return "unknown"


def make_query_event(
    text: str,
    locale: str = "ko",
    source_type: str = "manual",
) -> QueryEvent:
    """테스트 또는 API에서 QueryEvent를 편하게 만드는 팩토리."""
    return QueryEvent(
        query_id=str(uuid.uuid4()),
        text=text,
        occurred_at=datetime.now(tz=timezone.utc),
        locale=locale,
        source_type=source_type,
    )
