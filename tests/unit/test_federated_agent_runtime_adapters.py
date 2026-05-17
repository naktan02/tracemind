"""FL simulation agent runtime adapter 단위 검증."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from methods.federated_ssl.runtime_fallbacks import (
    RUNTIME_FALLBACK_TRAINING_PROFILE,
)
from scripts.runtime_adapters.federated_agent.backend_resolver import (
    resolve_example_generation_backend_name,
)
from scripts.runtime_adapters.federated_agent.row_validator import (
    require_rows_supported_by_example_backend,
)


def test_resolve_example_generation_backend_name_uses_runtime_fallback() -> None:
    assert resolve_example_generation_backend_name(objective_config=None) == (
        RUNTIME_FALLBACK_TRAINING_PROFILE.example_generation_backend_name
    )
    assert (
        resolve_example_generation_backend_name(
            objective_config=SimpleNamespace(example_generation_backend_name=None),
        )
        == RUNTIME_FALLBACK_TRAINING_PROFILE.example_generation_backend_name
    )


def test_resolve_example_generation_backend_name_prefers_objective_config() -> None:
    assert (
        resolve_example_generation_backend_name(
            objective_config=SimpleNamespace(
                example_generation_backend_name="weak_strong_pair",
            ),
        )
        == "weak_strong_pair"
    )


def test_row_validator_rejects_missing_weak_strong_views() -> None:
    with pytest.raises(
        ValueError,
        match="requires each row to include both weak_text/strong_text",
    ):
        require_rows_supported_by_example_backend(
            rows=[{"query_id": "q1", "text": "panic", "weak_text": "panic weak"}],
            backend_name="weak_strong_pair",
        )


def test_row_validator_accepts_usb_text_and_augmentation_fields() -> None:
    require_rows_supported_by_example_backend(
        rows=[
            {
                "query_id": "q1",
                "text": "panic weak",
                "aug_0": "panic strong de",
                "aug_1": "panic strong fr",
            }
        ],
        backend_name="weak_strong_pair",
    )


def test_row_validator_accepts_non_multiview_backend_without_view_fields() -> None:
    require_rows_supported_by_example_backend(
        rows=[{"query_id": "q1", "text": "panic"}],
        backend_name="prototype_rescore",
    )
