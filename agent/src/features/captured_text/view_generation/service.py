"""Captured text weak/strong view generation 서비스."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from agent.src.contracts.captured_text_contracts import (
    CapturedTextDebugJobRunResultPayload,
)
from agent.src.features.captured_text.storage.records import (
    CAPTURED_TEXT_VIEW_STATUS_FAILED,
    CAPTURED_TEXT_VIEW_STATUS_PENDING,
    CAPTURED_TEXT_VIEW_STATUS_READY,
    CapturedTextGeneratedViewRecord,
    CapturedTextRecord,
)
from agent.src.features.captured_text.storage.repository import (
    CapturedTextRepository,
)


class CapturedTextTranslationProvider(Protocol):
    """weak text 생성을 위한 번역 provider 경계."""

    def translate_batch(self, texts: list[str]) -> list[str]:
        """texts를 같은 순서의 target-language text로 변환한다."""
        ...


class CapturedTextStrongViewPair(Protocol):
    """strong view candidate pair shape."""

    aug_0: str
    aug_1: str


class CapturedTextStrongViewProvider(Protocol):
    """strong text 생성을 위한 provider 경계."""

    def build_candidate_pairs(
        self,
        *,
        texts: Sequence[str],
    ) -> list[CapturedTextStrongViewPair]:
        """texts를 같은 순서의 strong view pair로 변환한다."""
        ...


@dataclass(slots=True)
class CapturedTextViewGenerationService:
    """pending captured text를 weak/strong view로 materialize한다.

    실제 번역/augmentation backend는 app.state에서 주입되는 provider가 소유한다.
    provider가 없으면 개발 단계 fallback으로 identity view를 저장하고 metadata에
    그 사실을 남긴다.
    """

    repository: CapturedTextRepository
    translation_provider: CapturedTextTranslationProvider | None = None
    strong_view_provider: CapturedTextStrongViewProvider | None = None
    translation_locales: frozenset[str] = frozenset({"ko", "ja", "zh"})
    generator_name: str = "captured_text_view_generation"
    generator_version: str = "v1"

    @property
    def weak_text_provider_name(self) -> str:
        """debug/status용 weak text provider 이름."""

        return _provider_name(self.translation_provider)

    @property
    def strong_text_provider_name(self) -> str:
        """debug/status용 strong text provider 이름."""

        return _provider_name(self.strong_view_provider)

    @property
    def weak_text_identity_fallback(self) -> bool:
        """weak text가 identity fallback인지 여부."""

        return self.translation_provider is None

    @property
    def strong_text_identity_fallback(self) -> bool:
        """strong text가 identity fallback인지 여부."""

        return self.strong_view_provider is None

    def generate_pending_views(
        self,
        *,
        limit: int = 100,
    ) -> CapturedTextDebugJobRunResultPayload:
        """pending raw event를 읽어 generated view를 저장한다."""

        stale_count = self._reset_stale_ready_views(limit=limit)
        records = self.repository.get_pending_view_generation(limit=limit)
        if not records:
            return self._result(
                selected_count=0,
                generated_count=0,
                failed_count=0,
                message=_stale_view_message(stale_count),
            )

        try:
            weak_texts, weak_metadata = self._build_weak_texts(records)
            strong_pairs, strong_metadata = self._build_strong_texts(weak_texts)
        except Exception:
            for record in records:
                self.repository.mark_view_generation_status(
                    event_id=record.event_id,
                    status=CAPTURED_TEXT_VIEW_STATUS_FAILED,
                )
            return self._result(
                selected_count=len(records),
                generated_count=0,
                failed_count=len(records),
                message="view generation provider 실행 중 오류가 발생했습니다.",
            )

        generated_count = 0
        for index, record in enumerate(records):
            weak_text = weak_texts[index]
            strong_text_0, strong_text_1 = strong_pairs[index]
            if (
                not weak_text.strip()
                or not strong_text_0.strip()
                or not (strong_text_1.strip())
            ):
                self.repository.mark_view_generation_status(
                    event_id=record.event_id,
                    status=CAPTURED_TEXT_VIEW_STATUS_FAILED,
                )
                continue

            self.repository.save_generated_view(
                CapturedTextGeneratedViewRecord(
                    event_id=record.event_id,
                    generated_at=datetime.now(tz=timezone.utc),
                    weak_text=weak_text,
                    strong_text_0=strong_text_0,
                    strong_text_1=strong_text_1,
                    generator_name=self.generator_name,
                    generator_version=self.generator_version,
                    source_text_fingerprint=record.text_fingerprint,
                    metadata={
                        "source_locale": record.locale,
                        "source_type": record.source_type,
                        "surface_type": record.surface_type,
                        **weak_metadata[index],
                        **strong_metadata[index],
                    },
                )
            )
            self.repository.mark_view_generation_status(
                event_id=record.event_id,
                status=CAPTURED_TEXT_VIEW_STATUS_READY,
            )
            generated_count += 1

        failed_count = len(records) - generated_count
        return self._result(
            selected_count=len(records),
            generated_count=generated_count,
            failed_count=failed_count,
            message=_stale_view_message(stale_count),
        )

    def _reset_stale_ready_views(self, *, limit: int) -> int:
        """현재 provider와 맞지 않는 ready generated view를 재생성 대상으로 돌린다."""

        reset_count = 0
        for record in self.repository.get_recent_view_generation_by_status(
            status=CAPTURED_TEXT_VIEW_STATUS_READY,
            limit=limit,
        ):
            generated_view = self.repository.get_generated_view(record.event_id)
            if generated_view is None:
                reset_count += self.repository.mark_view_generation_status(
                    event_id=record.event_id,
                    status=CAPTURED_TEXT_VIEW_STATUS_PENDING,
                )
                continue
            if self._generated_view_matches_current_providers(
                record=record,
                generated_view=generated_view,
            ):
                continue
            self.repository.delete_generated_view(record.event_id)
            reset_count += self.repository.mark_view_generation_status(
                event_id=record.event_id,
                status=CAPTURED_TEXT_VIEW_STATUS_PENDING,
            )
        return reset_count

    def _generated_view_matches_current_providers(
        self,
        *,
        record: CapturedTextRecord,
        generated_view: CapturedTextGeneratedViewRecord,
    ) -> bool:
        metadata = generated_view.metadata
        expected_weak_provider = (
            self.weak_text_provider_name
            if self.translation_provider is not None
            and record.locale in self.translation_locales
            else "identity"
        )
        expected_strong_provider = self.strong_text_provider_name
        return (
            metadata.get("weak_text_provider") == expected_weak_provider
            and metadata.get("strong_text_provider") == expected_strong_provider
        )

    def _build_weak_texts(
        self,
        records: Sequence[CapturedTextRecord],
    ) -> tuple[list[str], list[dict[str, object]]]:
        texts = [record.text for record in records]
        metadata: list[dict[str, object]] = []
        weak_texts = list(texts)
        translatable_indexes = [
            index
            for index, record in enumerate(records)
            if record.locale in self.translation_locales
        ]
        if self.translation_provider is not None and translatable_indexes:
            translated = self.translation_provider.translate_batch(
                [texts[index] for index in translatable_indexes]
            )
            if len(translated) != len(translatable_indexes):
                raise ValueError("translation_provider returned mismatched count.")
            for offset, index in enumerate(translatable_indexes):
                weak_texts[index] = translated[offset]

        for index, record in enumerate(records):
            translated = (
                self.translation_provider is not None and index in translatable_indexes
            )
            metadata.append(
                {
                    "weak_text_role": "english_translation",
                    "weak_text_provider": (
                        _provider_name(self.translation_provider)
                        if translated
                        else "identity"
                    ),
                    "weak_text_translated": translated,
                    "weak_text_target_locale": "en" if translated else record.locale,
                }
            )
        return weak_texts, metadata

    def _build_strong_texts(
        self,
        weak_texts: Sequence[str],
    ) -> tuple[list[tuple[str, str]], list[dict[str, object]]]:
        if self.strong_view_provider is None:
            return (
                [(text, text) for text in weak_texts],
                [
                    {
                        "strong_text_provider": "identity",
                        "strong_text_augmented": False,
                    }
                    for _text in weak_texts
                ],
            )

        pairs = self.strong_view_provider.build_candidate_pairs(texts=weak_texts)
        if len(pairs) != len(weak_texts):
            raise ValueError("strong_view_provider returned mismatched count.")
        provider_name = _provider_name(self.strong_view_provider)
        return (
            [(pair.aug_0, pair.aug_1) for pair in pairs],
            [
                {
                    "strong_text_provider": provider_name,
                    "strong_text_augmented": True,
                }
                for _pair in pairs
            ],
        )

    def _result(
        self,
        *,
        selected_count: int,
        generated_count: int,
        failed_count: int,
        message: str = "",
    ) -> CapturedTextDebugJobRunResultPayload:
        counts = self.repository.count_by_view_generation_status()
        return CapturedTextDebugJobRunResultPayload(
            selected_count=selected_count,
            generated_count=generated_count,
            failed_count=failed_count,
            pending_remaining_count=counts.get(CAPTURED_TEXT_VIEW_STATUS_PENDING, 0),
            generated_view_count=self.repository.count_generated_views(),
            message=message,
        )


def _provider_name(provider: object | None) -> str:
    if provider is None:
        return "identity"
    adapter = getattr(provider, "adapter", None)
    model_id = getattr(adapter, "model_id", None)
    if isinstance(model_id, str) and model_id:
        return model_id
    provider_name = getattr(provider, "model_id", None)
    if isinstance(provider_name, str) and provider_name:
        return provider_name
    return provider.__class__.__name__


def _stale_view_message(stale_count: int) -> str:
    if stale_count <= 0:
        return ""
    return f"stale generated view {stale_count}건을 재생성했습니다."
