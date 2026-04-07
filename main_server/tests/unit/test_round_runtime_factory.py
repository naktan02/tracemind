"""server-owned round runtime config 조립 테스트."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import pytest

from main_server.src.api.main import create_app
from main_server.src.services.rounds import (
    ROUND_ADAPTER_FAMILY_ENV,
    ROUND_AGGREGATION_BACKEND_ENV,
    ServerRoundRuntimeConfig,
    SharedAdapterAggregationBackend,
    SharedAdapterRoundFamily,
    build_round_manager_service_from_config,
    build_shared_adapter_aggregation_backend,
    load_server_round_runtime_config_from_env,
    register_shared_adapter_aggregation_backend,
    register_shared_adapter_round_family,
)
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


def _build_test_round_family(aggregation_backend_name: str) -> SharedAdapterRoundFamily:
    return _TestRoundFamily(
        aggregation_backend=build_shared_adapter_aggregation_backend(
            adapter_kind=TEST_ADAPTER_KIND,
            backend_name=aggregation_backend_name,
        )
    )


register_shared_adapter_aggregation_backend(
    TEST_ADAPTER_KIND,
    TEST_BACKEND_NAME,
    factory=_TestAggregationBackend,
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


def test_round_runtime_config_rejects_incompatible_family_backend() -> None:
    with pytest.raises(ValueError):
        build_round_manager_service_from_config(
            ServerRoundRuntimeConfig(
                adapter_family_name="diagonal_scale",
                aggregation_backend_name=TEST_BACKEND_NAME,
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


def test_runtime_config_loader_reads_environment_mapping() -> None:
    config = load_server_round_runtime_config_from_env(
        environ={
            ROUND_ADAPTER_FAMILY_ENV: TEST_FAMILY_NAME,
            ROUND_AGGREGATION_BACKEND_ENV: TEST_BACKEND_NAME,
        }
    )

    assert config.adapter_family_name == TEST_FAMILY_NAME
    assert config.aggregation_backend_name == TEST_BACKEND_NAME


def test_main_server_app_uses_environment_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ROUND_ADAPTER_FAMILY_ENV, TEST_FAMILY_NAME)
    monkeypatch.setenv(ROUND_AGGREGATION_BACKEND_ENV, TEST_BACKEND_NAME)

    app = create_app()
    config = app.state.round_runtime_config
    service = app.state.round_lifecycle_service

    assert config.adapter_family_name == TEST_FAMILY_NAME
    assert config.aggregation_backend_name == TEST_BACKEND_NAME
    assert isinstance(service.round_manager_service.adapter_family, _TestRoundFamily)
