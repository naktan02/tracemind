"""Captured text view provider env wiring tests."""

from __future__ import annotations

from pathlib import Path

from agent.src.infrastructure.repositories.captured_text.repository import (
    CapturedTextRepository,
)
from agent.src.services.captured_text.view_generation.provider_factory import (
    build_captured_text_view_generation_service_from_env,
    load_captured_text_view_provider_config,
)


def test_provider_factory_defaults_to_identity_fallback(tmp_path: Path) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")

    service = build_captured_text_view_generation_service_from_env(
        repository=repository,
        environ={},
    )

    assert service.weak_text_provider_name == "identity"
    assert service.strong_text_provider_name == "identity"
    assert service.weak_text_identity_fallback is True
    assert service.strong_text_identity_fallback is True


def test_provider_factory_wires_nllb_services_without_loading_model(
    tmp_path: Path,
) -> None:
    repository = CapturedTextRepository(db_path=tmp_path / "captured_text.db")

    service = build_captured_text_view_generation_service_from_env(
        repository=repository,
        environ={
            "TRACEMIND_CAPTURED_TEXT_TRANSLATION_PROVIDER": "nllb",
            "TRACEMIND_CAPTURED_TEXT_STRONG_VIEW_PROVIDER": "nllb_backtranslation",
            "TRACEMIND_CAPTURED_TEXT_TRANSLATION_SOURCE_LOCALE": "ko",
            "TRACEMIND_CAPTURED_TEXT_NLLB_LOCAL_FILES_ONLY": "true",
        },
    )

    assert service.weak_text_provider_name == "facebook/nllb-200-distilled-600M"
    assert service.strong_text_provider_name == "facebook/nllb-200-distilled-600M"
    assert service.weak_text_identity_fallback is False
    assert service.strong_text_identity_fallback is False
    assert service.translation_locales == frozenset({"ko"})


def test_provider_config_rejects_unknown_provider() -> None:
    try:
        load_captured_text_view_provider_config(
            environ={"TRACEMIND_CAPTURED_TEXT_TRANSLATION_PROVIDER": "unknown"}
        )
    except ValueError as error:
        assert "TRACEMIND_CAPTURED_TEXT_TRANSLATION_PROVIDER" in str(error)
    else:
        raise AssertionError("unknown provider should fail")
