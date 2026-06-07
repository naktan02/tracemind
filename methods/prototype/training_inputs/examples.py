"""Prototype score 기반 training input view 계산 core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    extract_category_prototypes,
)
from shared.src.domain.entities.inference.events import AnalysisEvent

PrototypeMapping = Mapping[str, Sequence[float] | Sequence[Sequence[float]]]


class PrototypeTrainingInputSource(Protocol):
    """Prototype training input core가 읽는 source row 최소 shape."""

    query_id: str
    occurred_at: datetime
    translated_text: str | None
    weak_text: str | None
    strong_text: str | None
    weak_translated_text: str | None
    strong_translated_text: str | None


class StoredPrototypeAnalysisEvent(Protocol):
    """Stored analysis event 재점수화에 필요한 최소 shape."""

    analysis_event: AnalysisEvent
    base_embedding: Sequence[float] | None


class PrototypeEmbeddingTransform(Protocol):
    """Shared adapter state처럼 embedding을 변환하는 객체."""

    def apply(self, embedding: Sequence[float]) -> Sequence[float]:
        """base embedding에 adapter transform을 적용한다."""


class PrototypeScoreBackend(Protocol):
    """Prototype score dict를 계산하는 scorer 최소 interface."""

    def score(
        self,
        embedding: Sequence[float],
        prototypes: PrototypeMapping,
    ) -> dict[str, float]:
        """embedding과 prototypes로 category score dict를 계산한다."""


@dataclass(frozen=True, slots=True)
class PrototypeSingleViewTrainingInput:
    """단일 view prototype 재점수화 결과."""

    analysis_event: AnalysisEvent
    embedding: list[float]
    base_embedding: list[float]


@dataclass(frozen=True, slots=True)
class PrototypeWeakStrongTrainingInput:
    """weak selection view와 strong update view를 가진 prototype input."""

    weak_analysis_event: AnalysisEvent
    strong_analysis_event: AnalysisEvent
    weak_embedding: list[float]
    strong_embedding: list[float]
    weak_base_embedding: list[float]
    strong_base_embedding: list[float]
    metadata: dict[str, str]


def build_prototype_rescore_inputs(
    *,
    source_rows: Sequence[PrototypeTrainingInputSource],
    base_embeddings: Sequence[Sequence[float]],
    adapter_state: PrototypeEmbeddingTransform,
    prototype_pack: PrototypePackPayload,
    model_id: str,
    scorer: PrototypeScoreBackend,
) -> tuple[PrototypeSingleViewTrainingInput, ...]:
    """source row와 base embedding으로 single-view input을 만든다."""

    if not source_rows:
        return ()

    prototypes = extract_category_prototypes(prototype_pack)
    inputs: list[PrototypeSingleViewTrainingInput] = []
    for row, base_embedding in zip(source_rows, base_embeddings, strict=True):
        adapted_embedding = list(adapter_state.apply(base_embedding))
        analysis_event = AnalysisEvent(
            query_id=row.query_id,
            occurred_at=row.occurred_at,
            translated_text=row.translated_text,
            embedding_model_id=model_id,
            translation_model_id=prototype_pack.translation_model_id,
            category_scores=scorer.score(adapted_embedding, prototypes),
        )
        inputs.append(
            PrototypeSingleViewTrainingInput(
                analysis_event=analysis_event,
                embedding=adapted_embedding,
                base_embedding=list(base_embedding),
            )
        )
    return tuple(inputs)


def build_prototype_rescore_inputs_from_stored_events(
    *,
    stored_events: Sequence[StoredPrototypeAnalysisEvent],
    adapter_state: PrototypeEmbeddingTransform,
    prototype_pack: PrototypePackPayload,
    scorer: PrototypeScoreBackend,
) -> tuple[PrototypeSingleViewTrainingInput, ...]:
    """저장된 analysis event를 현재 prototype/adapter 기준으로 재점수화한다."""

    prototypes = extract_category_prototypes(prototype_pack)
    inputs: list[PrototypeSingleViewTrainingInput] = []
    for stored_event in stored_events:
        base_embedding = stored_event.base_embedding
        if base_embedding is None or len(base_embedding) == 0:
            continue
        adapted_embedding = list(adapter_state.apply(base_embedding))
        analysis_event = replace(
            stored_event.analysis_event,
            category_scores=scorer.score(adapted_embedding, prototypes),
        )
        inputs.append(
            PrototypeSingleViewTrainingInput(
                analysis_event=analysis_event,
                embedding=adapted_embedding,
                base_embedding=list(base_embedding),
            )
        )
    return tuple(inputs)


def require_weak_strong_texts(
    source_rows: Sequence[PrototypeTrainingInputSource],
) -> tuple[list[str], list[str]]:
    """weak/strong text가 모두 있는지 검증하고 embedding 입력 text를 반환한다."""

    weak_texts: list[str] = []
    strong_texts: list[str] = []
    for row in source_rows:
        if row.weak_text is None or row.strong_text is None:
            raise ValueError(
                "weak_strong_pair backend requires both weak_text and "
                "strong_text for every TrainingExampleSource."
            )
        weak_texts.append(row.weak_text)
        strong_texts.append(row.strong_text)
    return weak_texts, strong_texts


def build_prototype_weak_strong_inputs(
    *,
    source_rows: Sequence[PrototypeTrainingInputSource],
    weak_base_embeddings: Sequence[Sequence[float]],
    strong_base_embeddings: Sequence[Sequence[float]],
    adapter_state: PrototypeEmbeddingTransform,
    prototype_pack: PrototypePackPayload,
    model_id: str,
    scorer: PrototypeScoreBackend,
    backend_name: str,
) -> tuple[PrototypeWeakStrongTrainingInput, ...]:
    """weak view는 selection, strong view는 update에 쓰는 multiview input을 만든다."""

    if not source_rows:
        return ()

    prototypes = extract_category_prototypes(prototype_pack)
    inputs: list[PrototypeWeakStrongTrainingInput] = []
    for row, weak_base_embedding, strong_base_embedding in zip(
        source_rows,
        weak_base_embeddings,
        strong_base_embeddings,
        strict=True,
    ):
        weak_embedding = list(adapter_state.apply(weak_base_embedding))
        strong_embedding = list(adapter_state.apply(strong_base_embedding))
        weak_analysis_event = AnalysisEvent(
            query_id=row.query_id,
            occurred_at=row.occurred_at,
            translated_text=row.weak_translated_text or row.translated_text,
            embedding_model_id=model_id,
            translation_model_id=prototype_pack.translation_model_id,
            category_scores=scorer.score(weak_embedding, prototypes),
        )
        strong_analysis_event = AnalysisEvent(
            query_id=row.query_id,
            occurred_at=row.occurred_at,
            translated_text=row.strong_translated_text or row.translated_text,
            embedding_model_id=model_id,
            translation_model_id=prototype_pack.translation_model_id,
            category_scores=scorer.score(strong_embedding, prototypes),
        )
        inputs.append(
            PrototypeWeakStrongTrainingInput(
                weak_analysis_event=weak_analysis_event,
                strong_analysis_event=strong_analysis_event,
                weak_embedding=weak_embedding,
                strong_embedding=strong_embedding,
                weak_base_embedding=list(weak_base_embedding),
                strong_base_embedding=list(strong_base_embedding),
                metadata={
                    "training_input_backend_name": backend_name,
                    "selection_view": "weak",
                    "update_view": "strong",
                },
            )
        )
    return tuple(inputs)
