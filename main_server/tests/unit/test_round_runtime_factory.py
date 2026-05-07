"""server-owned round runtime config 조립 테스트."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import pytest
from fastapi.middleware.cors import CORSMiddleware

from main_server.src.api.main import (
    DEFAULT_EXPERIMENT_WEB_ALLOWED_ORIGINS,
    create_app,
    load_experiment_web_allowed_origins_from_env,
)
from main_server.src.services.federation.rounds.aggregation.models import (
    SharedAdapterAggregationBackend,
)
from main_server.src.services.federation.rounds.aggregation.registry import (
    build_shared_adapter_aggregation_backend,
    register_shared_adapter_aggregation_backend,
)
from main_server.src.services.federation.rounds.families.models import (
    SharedAdapterRoundFamily,
)
from main_server.src.services.federation.rounds.families.registry import (
    register_shared_adapter_round_family,
)
from main_server.src.services.federation.rounds.runtime.config import (
    ROUND_ADAPTER_FAMILY_ENV,
    ROUND_AGGREGATION_BACKEND_CONFIG_ENV,
    ROUND_AGGREGATION_BACKEND_ENV,
    ServerRoundRuntimeConfig,
    load_server_round_runtime_config_from_env,
)
from main_server.src.services.federation.rounds.runtime.factory import (
    build_round_manager_service_from_config,
)
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.adapter_contracts import (
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

TEST_ADAPTER_KIND = "test_adapter_runtime_factory"
TEST_FAMILY_NAME = "test_family_runtime_factory"
TEST_BACKEND_NAME = "test_avg_runtime_factory"
TEST_MISMATCH_BACKEND_NAME = "test_mismatch_avg_runtime_factory"


@dataclass(slots=True)
class _TestAggregationBackend:
    adapter_kind: str = TEST_ADAPTER_KIND

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        next_model_revision: str,
        aggregated_at: datetime,
    ):
        raise NotImplementedError


@dataclass(slots=True)
class _MismatchedAggregationBackend:
    adapter_kind: str = "wrong_adapter_kind"

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        next_model_revision: str,
        aggregated_at: datetime,
    ):
        raise NotImplementedError


@dataclass(slots=True)
class _TestRoundFamily:
    adapter_kind: str = TEST_ADAPTER_KIND
    accepted_update_formats: tuple[str, ...] = ("test_update",)
    aggregation_backend: SharedAdapterAggregationBackend | None = None

    def state_from_payload(
        self,
        payload: SharedAdapterStatePayload,
    ) -> SharedAdapterState:
        raise NotImplementedError

    def update_from_payload(
        self,
        payload: SharedAdapterUpdatePayload,
    ) -> SharedAdapterUpdate:
        raise NotImplementedError

    def state_to_payload(
        self,
        state: SharedAdapterState,
    ) -> SharedAdapterStatePayload:
        raise NotImplementedError


def _build_test_round_family(
    aggregation_backend_name: str,
    aggregation_backend_overrides,
) -> SharedAdapterRoundFamily:
    del aggregation_backend_overrides
    return _TestRoundFamily(
        aggregation_backend=build_shared_adapter_aggregation_backend(
            adapter_kind=TEST_ADAPTER_KIND,
            backend_name=aggregation_backend_name,
        )
    )


register_shared_adapter_aggregation_backend(
    TEST_ADAPTER_KIND,
    TEST_BACKEND_NAME,
    factory=lambda _overrides: _TestAggregationBackend(),
    catalog_entry=RegistryCatalogEntry(
        item_name=f"{TEST_ADAPTER_KIND}.{TEST_BACKEND_NAME}",
        display_name=TEST_BACKEND_NAME,
        implementation_module=__name__,
        core_method_name=TEST_BACKEND_NAME,
        family_name=TEST_ADAPTER_KIND,
        supported_adapter_kinds=(TEST_ADAPTER_KIND,),
    ),
)
register_shared_adapter_aggregation_backend(
    TEST_ADAPTER_KIND,
    TEST_MISMATCH_BACKEND_NAME,
    factory=lambda _overrides: _MismatchedAggregationBackend(),
    catalog_entry=RegistryCatalogEntry(
        item_name=f"{TEST_ADAPTER_KIND}.{TEST_MISMATCH_BACKEND_NAME}",
        display_name=TEST_MISMATCH_BACKEND_NAME,
        implementation_module=__name__,
        core_method_name=TEST_MISMATCH_BACKEND_NAME,
        family_name=TEST_ADAPTER_KIND,
        supported_adapter_kinds=(TEST_ADAPTER_KIND,),
    ),
)
register_shared_adapter_round_family(
    TEST_FAMILY_NAME,
    factory=_build_test_round_family,
)


def test_round_runtime_config_builds_registered_family_and_backend() -> None:
    service = build_round_manager_service_from_config(
        ServerRoundRuntimeConfig(
            adapter_family_name=TEST_FAMILY_NAME,
            aggregation_backend_name=TEST_BACKEND_NAME,
        )
    )

    assert isinstance(service.adapter_family, _TestRoundFamily)
    assert isinstance(
        service.adapter_family.aggregation_backend,
        _TestAggregationBackend,
    )


def test_round_runtime_config_builds_lora_classifier_family() -> None:
    service = build_round_manager_service_from_config(
        ServerRoundRuntimeConfig(
            adapter_family_name="lora_classifier",
            aggregation_backend_name="fedavg",
        )
    )

    assert service.adapter_family.adapter_kind == "lora_classifier"
    assert service.adapter_family.aggregation_backend.adapter_kind == "lora_classifier"
    assert service.adapter_family.accepted_update_formats == ("lora_classifier_update",)


def test_round_runtime_config_rejects_incompatible_family_backend() -> None:
    with pytest.raises(ValueError):
        build_round_manager_service_from_config(
            ServerRoundRuntimeConfig(
                adapter_family_name="diagonal_scale",
                aggregation_backend_name=TEST_BACKEND_NAME,
            )
        )


def test_round_runtime_config_rejects_mismatched_backend_adapter_kind() -> None:
    with pytest.raises(ValueError, match="Incompatible round runtime config"):
        build_round_manager_service_from_config(
            ServerRoundRuntimeConfig(
                adapter_family_name=TEST_FAMILY_NAME,
                aggregation_backend_name=TEST_MISMATCH_BACKEND_NAME,
            )
        )


def test_main_server_app_uses_runtime_config_to_build_round_service() -> None:
    app = create_app(
        round_runtime_config=ServerRoundRuntimeConfig(
            adapter_family_name=TEST_FAMILY_NAME,
            aggregation_backend_name=TEST_BACKEND_NAME,
        )
    )

    service = app.state.round_lifecycle_service
    assert isinstance(service.round_manager_service.adapter_family, _TestRoundFamily)
    assert (
        service.round_manager_service.adapter_family.aggregation_backend.adapter_kind
        == TEST_ADAPTER_KIND
    )
    cors_middleware = next(
        middleware
        for middleware in app.user_middleware
        if middleware.cls is CORSMiddleware
    )
    assert cors_middleware.kwargs["allow_origins"] == list(
        DEFAULT_EXPERIMENT_WEB_ALLOWED_ORIGINS
    )


def test_runtime_config_loader_reads_environment_mapping() -> None:
    config = load_server_round_runtime_config_from_env(
        environ={
            ROUND_ADAPTER_FAMILY_ENV: TEST_FAMILY_NAME,
            ROUND_AGGREGATION_BACKEND_ENV: TEST_BACKEND_NAME,
            ROUND_AGGREGATION_BACKEND_CONFIG_ENV: '{"min_scale": 0.8}',
        }
    )

    assert config.adapter_family_name == TEST_FAMILY_NAME
    assert config.aggregation_backend_name == TEST_BACKEND_NAME
    assert config.aggregation_backend_overrides == {"min_scale": 0.8}


def test_experiment_web_origin_loader_reads_environment_mapping() -> None:
    origins = load_experiment_web_allowed_origins_from_env(
        environ={
            "EXPERIMENT_WEB_ALLOWED_ORIGINS": (
                "http://localhost:5173, https://experiment.example.com "
            )
        }
    )

    assert origins == (
        "http://localhost:5173",
        "https://experiment.example.com",
    )


def test_main_server_app_uses_environment_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ROUND_ADAPTER_FAMILY_ENV, TEST_FAMILY_NAME)
    monkeypatch.setenv(ROUND_AGGREGATION_BACKEND_ENV, TEST_BACKEND_NAME)
    monkeypatch.setenv(ROUND_AGGREGATION_BACKEND_CONFIG_ENV, '{"max_scale": 1.1}')

    app = create_app()
    config = app.state.round_runtime_config
    service = app.state.round_lifecycle_service

    assert config.adapter_family_name == TEST_FAMILY_NAME
    assert config.aggregation_backend_name == TEST_BACKEND_NAME
    assert config.aggregation_backend_overrides == {"max_scale": 1.1}
    assert isinstance(service.round_manager_service.adapter_family, _TestRoundFamily)
