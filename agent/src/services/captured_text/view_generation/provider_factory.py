"""Captured text view generation provider env wiring."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from agent.src.infrastructure.model_adapters.translation.nllb import (
    NllbTranslationAdapter,
)
from agent.src.infrastructure.repositories.captured_text.repository import (
    CapturedTextRepository,
)
from agent.src.services.captured_text.view_generation.service import (
    CapturedTextViewGenerationService,
)
from agent.src.services.language.backtranslation_service import (
    NllbBacktranslationService,
)
from agent.src.services.language.translation_service import TranslationService

CAPTURED_TEXT_TRANSLATION_PROVIDER_ENV = "TRACEMIND_CAPTURED_TEXT_TRANSLATION_PROVIDER"
CAPTURED_TEXT_TRANSLATION_SOURCE_LOCALE_ENV = (
    "TRACEMIND_CAPTURED_TEXT_TRANSLATION_SOURCE_LOCALE"
)
CAPTURED_TEXT_TRANSLATION_SOURCE_LANG_ENV = (
    "TRACEMIND_CAPTURED_TEXT_TRANSLATION_SOURCE_LANG"
)
CAPTURED_TEXT_TRANSLATION_TARGET_LANG_ENV = (
    "TRACEMIND_CAPTURED_TEXT_TRANSLATION_TARGET_LANG"
)
CAPTURED_TEXT_STRONG_VIEW_PROVIDER_ENV = "TRACEMIND_CAPTURED_TEXT_STRONG_VIEW_PROVIDER"
CAPTURED_TEXT_STRONG_SOURCE_LANG_ENV = "TRACEMIND_CAPTURED_TEXT_STRONG_SOURCE_LANG"
CAPTURED_TEXT_STRONG_PIVOT_LANGS_ENV = "TRACEMIND_CAPTURED_TEXT_STRONG_PIVOT_LANGS"
CAPTURED_TEXT_NLLB_MODEL_ID_ENV = "TRACEMIND_CAPTURED_TEXT_NLLB_MODEL_ID"
CAPTURED_TEXT_NLLB_REVISION_ENV = "TRACEMIND_CAPTURED_TEXT_NLLB_REVISION"
CAPTURED_TEXT_NLLB_DEVICE_ENV = "TRACEMIND_CAPTURED_TEXT_NLLB_DEVICE"
CAPTURED_TEXT_NLLB_BATCH_SIZE_ENV = "TRACEMIND_CAPTURED_TEXT_NLLB_BATCH_SIZE"
CAPTURED_TEXT_NLLB_MAX_NEW_TOKENS_ENV = "TRACEMIND_CAPTURED_TEXT_NLLB_MAX_NEW_TOKENS"
CAPTURED_TEXT_NLLB_TORCH_DTYPE_ENV = "TRACEMIND_CAPTURED_TEXT_NLLB_TORCH_DTYPE"
CAPTURED_TEXT_NLLB_CACHE_DIR_ENV = "TRACEMIND_CAPTURED_TEXT_NLLB_CACHE_DIR"
CAPTURED_TEXT_NLLB_LOCAL_FILES_ONLY_ENV = (
    "TRACEMIND_CAPTURED_TEXT_NLLB_LOCAL_FILES_ONLY"
)

DEFAULT_NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
DEFAULT_NLLB_REVISION = "main"
DEFAULT_TRANSLATION_SOURCE_LOCALE = "ko"
DEFAULT_TRANSLATION_SOURCE_LANG = "kor_Hang"
DEFAULT_TRANSLATION_TARGET_LANG = "eng_Latn"
DEFAULT_STRONG_SOURCE_LANG = "eng_Latn"
DEFAULT_STRONG_PIVOT_LANGS = ("fra_Latn", "deu_Latn")


@dataclass(frozen=True, slots=True)
class CapturedTextViewProviderConfig:
    """captured text view generation provider 설정."""

    translation_provider_name: str
    translation_source_locale: str
    translation_source_lang: str
    translation_target_lang: str
    strong_view_provider_name: str
    strong_source_lang: str
    strong_pivot_langs: tuple[str, str]
    nllb_model_id: str
    nllb_revision: str
    nllb_device: str
    nllb_batch_size: int
    nllb_max_new_tokens: int
    nllb_torch_dtype: str
    nllb_cache_dir: str | None
    nllb_local_files_only: bool


def build_captured_text_view_generation_service_from_env(
    *,
    repository: CapturedTextRepository,
    environ: Mapping[str, str] | None = None,
) -> CapturedTextViewGenerationService:
    """환경변수에서 captured text view generation service를 조립한다."""

    config = load_captured_text_view_provider_config(environ=environ)
    translation_provider = None
    translation_locales = frozenset({config.translation_source_locale})
    if config.translation_provider_name == "nllb":
        translation_provider = TranslationService(
            adapter=NllbTranslationAdapter(
                model_id=config.nllb_model_id,
                revision=config.nllb_revision,
                source_lang=config.translation_source_lang,
                target_lang=config.translation_target_lang,
                device=config.nllb_device,
                batch_size=config.nllb_batch_size,
                max_new_tokens=config.nllb_max_new_tokens,
                torch_dtype=config.nllb_torch_dtype,
                cache_dir=config.nllb_cache_dir,
                local_files_only=config.nllb_local_files_only,
            )
        )

    strong_view_provider = None
    if config.strong_view_provider_name == "nllb_backtranslation":
        strong_view_provider = NllbBacktranslationService(
            source_lang=config.strong_source_lang,
            pivot_languages=config.strong_pivot_langs,
            model_id=config.nllb_model_id,
            revision=config.nllb_revision,
            device=config.nllb_device,
            batch_size=config.nllb_batch_size,
            max_new_tokens=config.nllb_max_new_tokens,
            torch_dtype=config.nllb_torch_dtype,
            cache_dir=config.nllb_cache_dir,
            local_files_only=config.nllb_local_files_only,
        )

    return CapturedTextViewGenerationService(
        repository=repository,
        translation_provider=translation_provider,
        strong_view_provider=strong_view_provider,
        translation_locales=translation_locales,
    )


def load_captured_text_view_provider_config(
    *,
    environ: Mapping[str, str] | None = None,
) -> CapturedTextViewProviderConfig:
    """환경변수에서 provider config를 정규화한다."""

    effective_environ = os.environ if environ is None else environ
    translation_provider_name = _provider_name(
        effective_environ.get(CAPTURED_TEXT_TRANSLATION_PROVIDER_ENV, "none")
    )
    if translation_provider_name not in {"none", "identity", "nllb"}:
        raise ValueError(
            "TRACEMIND_CAPTURED_TEXT_TRANSLATION_PROVIDER는 "
            "'none', 'identity', 'nllb' 중 하나여야 합니다."
        )
    strong_view_provider_name = _provider_name(
        effective_environ.get(CAPTURED_TEXT_STRONG_VIEW_PROVIDER_ENV, "none")
    )
    if strong_view_provider_name not in {"none", "identity", "nllb_backtranslation"}:
        raise ValueError(
            "TRACEMIND_CAPTURED_TEXT_STRONG_VIEW_PROVIDER는 "
            "'none', 'identity', 'nllb_backtranslation' 중 하나여야 합니다."
        )

    return CapturedTextViewProviderConfig(
        translation_provider_name=translation_provider_name,
        translation_source_locale=_env_value(
            effective_environ,
            CAPTURED_TEXT_TRANSLATION_SOURCE_LOCALE_ENV,
            DEFAULT_TRANSLATION_SOURCE_LOCALE,
        ).lower(),
        translation_source_lang=_env_value(
            effective_environ,
            CAPTURED_TEXT_TRANSLATION_SOURCE_LANG_ENV,
            DEFAULT_TRANSLATION_SOURCE_LANG,
        ),
        translation_target_lang=_env_value(
            effective_environ,
            CAPTURED_TEXT_TRANSLATION_TARGET_LANG_ENV,
            DEFAULT_TRANSLATION_TARGET_LANG,
        ),
        strong_view_provider_name=strong_view_provider_name,
        strong_source_lang=_env_value(
            effective_environ,
            CAPTURED_TEXT_STRONG_SOURCE_LANG_ENV,
            DEFAULT_STRONG_SOURCE_LANG,
        ),
        strong_pivot_langs=_parse_pivot_langs(
            effective_environ.get(CAPTURED_TEXT_STRONG_PIVOT_LANGS_ENV, "")
        ),
        nllb_model_id=_env_value(
            effective_environ,
            CAPTURED_TEXT_NLLB_MODEL_ID_ENV,
            DEFAULT_NLLB_MODEL_ID,
        ),
        nllb_revision=_env_value(
            effective_environ,
            CAPTURED_TEXT_NLLB_REVISION_ENV,
            DEFAULT_NLLB_REVISION,
        ),
        nllb_device=_env_value(
            effective_environ, CAPTURED_TEXT_NLLB_DEVICE_ENV, "auto"
        ),
        nllb_batch_size=_env_int(
            effective_environ,
            CAPTURED_TEXT_NLLB_BATCH_SIZE_ENV,
            8,
        ),
        nllb_max_new_tokens=_env_int(
            effective_environ,
            CAPTURED_TEXT_NLLB_MAX_NEW_TOKENS_ENV,
            256,
        ),
        nllb_torch_dtype=_env_value(
            effective_environ,
            CAPTURED_TEXT_NLLB_TORCH_DTYPE_ENV,
            "auto",
        ),
        nllb_cache_dir=_env_optional_value(
            effective_environ,
            CAPTURED_TEXT_NLLB_CACHE_DIR_ENV,
        ),
        nllb_local_files_only=_env_bool(
            effective_environ,
            CAPTURED_TEXT_NLLB_LOCAL_FILES_ONLY_ENV,
            False,
        ),
    )


def _provider_name(value: str | None) -> str:
    normalized = str(value or "none").strip().lower()
    return "none" if normalized == "" else normalized


def _env_value(environ: Mapping[str, str], key: str, default: str) -> str:
    value = environ.get(key, "").strip()
    return value or default


def _env_optional_value(environ: Mapping[str, str], key: str) -> str | None:
    value = environ.get(key, "").strip()
    return value or None


def _env_int(environ: Mapping[str, str], key: str, default: int) -> int:
    value = environ.get(key, "").strip()
    return int(value) if value else default


def _env_bool(environ: Mapping[str, str], key: str, default: bool) -> bool:
    value = environ.get(key, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be boolean-like.")


def _parse_pivot_langs(raw_value: str) -> tuple[str, str]:
    values = tuple(value.strip() for value in raw_value.split(",") if value.strip())
    if not values:
        return DEFAULT_STRONG_PIVOT_LANGS
    if len(values) != 2:
        raise ValueError(
            "TRACEMIND_CAPTURED_TEXT_STRONG_PIVOT_LANGS는 쉼표로 구분한 "
            "정확히 두 언어 코드여야 합니다."
        )
    return (values[0], values[1])
