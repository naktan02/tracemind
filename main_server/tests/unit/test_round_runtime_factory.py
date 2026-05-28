"""server-owned round runtime config 조립 테스트."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import pytest

from main_server.src.api.main import create_app
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
from main_server.src.services.federation.rounds.runtime import (
    compatibility as runtime_compatibility_module,
)
from main_server.src.services.federation.rounds.runtime import (
    factory as runtime_factory_module,
)
from main_server.src.services.federation.rounds.runtime.config import (
    ROUND_AGGREGATION_BACKEND_CONFIG_ENV,
    ROUND_AGGREGATION_BACKEND_ENV,
    ROUND_METHOD_DESCRIPTOR_ENV,
    ROUND_PAYLOAD_ADAPTER_KIND_ENV,
    ROUND_UPDATE_FAMILY_ENV,
    ServerRoundRuntimeConfig,
    load_server_round_runtime_config_from_env,
)
from main_server.src.services.federation.rounds.runtime.factory import (
    build_round_lifecycle_service_from_config,
    build_round_manager_service_from_config,
)
from methods.federated_ssl.base import (
    FederatedSslLocalStepSpec,
    FederatedSslMethodDescriptor,
    FederatedSslMethodRecipe,
    FederatedSslRequiredViews,
    FederatedSslRuntimeCapabilities,
    FederatedSslRuntimePair,
    FederatedSslServerStepSpec,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

TEST_ADAPTER_KIND = "test_adapter_runtime_factory"
TEST_FAMILY_NAME = "test_family_runtime_factory"
TEST_UPDATE_FAMILY_NAME = "test_update_family_runtime_factory"
TEST_BACKEND_NAME = "test_avg_runtime_factory"
TEST_MISMATCH_BACKEND_NAME = "test_mismatch_avg_runtime_factory"
TEST_METHOD_NAME = "test_round_runtime_descriptor"
TEST_METHOD_DESCRIPTOR = FederatedSslMethodDescriptor(
    name=TEST_METHOD_NAME,
    implementation_status="test_only",
    required_views=FederatedSslRequiredViews(
        view_names=("single_view",),
        view_generator_name="training_example_backend",
    ),
    local_step=FederatedSslLocalStepSpec(
        step_name="pseudo_label_self_training",
        client_trainer_name="local_training_service",
        pseudo_labeler_name="ssl_pseudo_label_selection_hook",
    ),
    server_step=FederatedSslServerStepSpec(
        server_aggregator_name="round_runtime_aggregation_backend",
        round_policy_name="round_active_pair_only",
        server_aggregate_hint="use_round_runtime_aggregation_backend",
    ),
    runtime_capabilities=FederatedSslRuntimeCapabilities(
        simulation_supported=True,
        live_agent_supported=True,
        live_server_supported=True,
    ),
    recipe=FederatedSslMethodRecipe(
        method_name=TEST_METHOD_NAME,
        supported_runtime_pairs=(
            FederatedSslRuntimePair(
                update_family_name=TEST_UPDATE_FAMILY_NAME,
                aggregation_backend_name=TEST_BACKEND_NAME,
            ),
        ),
    ),
)


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
            payload_adapter_kind=TEST_FAMILY_NAME,
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
            payload_adapter_kind="lora_classifier",
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
                payload_adapter_kind="peft_classifier",
                aggregation_backend_name=TEST_BACKEND_NAME,
            )
        )


def test_round_runtime_config_rejects_mismatched_backend_adapter_kind() -> None:
    with pytest.raises(ValueError, match="Incompatible round runtime config"):
        build_round_manager_service_from_config(
            ServerRoundRuntimeConfig(
                payload_adapter_kind=TEST_FAMILY_NAME,
                aggregation_backend_name=TEST_MISMATCH_BACKEND_NAME,
            )
        )


def test_round_runtime_config_rejects_method_recipe_runtime_pair_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_compatibility_module,
        "resolve_federated_ssl_method_descriptor",
        lambda _name: TEST_METHOD_DESCRIPTOR,
    )
    with pytest.raises(ValueError, match="method recipe does not support"):
        build_round_manager_service_from_config(
            ServerRoundRuntimeConfig(
                payload_adapter_kind="lora_classifier",
                aggregation_backend_name="fedavg",
                method_descriptor_name=TEST_METHOD_NAME,
            )
        )


def test_round_lifecycle_config_wires_method_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_compatibility_module,
        "resolve_federated_ssl_method_descriptor",
        lambda _name: TEST_METHOD_DESCRIPTOR,
    )
    monkeypatch.setattr(
        runtime_factory_module,
        "resolve_federated_ssl_method_descriptor",
        lambda _name: TEST_METHOD_DESCRIPTOR,
    )
    service = build_round_lifecycle_service_from_config(
        ServerRoundRuntimeConfig(
            payload_adapter_kind=TEST_FAMILY_NAME,
            update_family_name=TEST_UPDATE_FAMILY_NAME,
            aggregation_backend_name=TEST_BACKEND_NAME,
            method_descriptor_name=TEST_METHOD_NAME,
        )
    )

    assert service.method_descriptor is not None
    assert service.method_descriptor.name == TEST_METHOD_NAME


def test_main_server_app_uses_runtime_config_to_build_round_service() -> None:
    app = create_app(
        round_runtime_config=ServerRoundRuntimeConfig(
            payload_adapter_kind=TEST_FAMILY_NAME,
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
            ROUND_PAYLOAD_ADAPTER_KIND_ENV: TEST_FAMILY_NAME,
            ROUND_UPDATE_FAMILY_ENV: TEST_UPDATE_FAMILY_NAME,
            ROUND_AGGREGATION_BACKEND_ENV: TEST_BACKEND_NAME,
            ROUND_METHOD_DESCRIPTOR_ENV: f" {TEST_METHOD_NAME} ",
            ROUND_AGGREGATION_BACKEND_CONFIG_ENV: '{"min_scale": 0.8}',
        }
    )

    assert config.payload_adapter_kind == TEST_FAMILY_NAME
    assert config.update_family_name == TEST_UPDATE_FAMILY_NAME
    assert config.aggregation_backend_name == TEST_BACKEND_NAME
    assert config.method_descriptor_name == TEST_METHOD_NAME
    assert config.aggregation_backend_overrides == {"min_scale": 0.8}


def test_main_server_app_uses_environment_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ROUND_PAYLOAD_ADAPTER_KIND_ENV, TEST_FAMILY_NAME)
    monkeypatch.setenv(ROUND_UPDATE_FAMILY_ENV, TEST_UPDATE_FAMILY_NAME)
    monkeypatch.setenv(ROUND_AGGREGATION_BACKEND_ENV, TEST_BACKEND_NAME)
    monkeypatch.setenv(ROUND_AGGREGATION_BACKEND_CONFIG_ENV, '{"max_scale": 1.1}')

    app = create_app()
    config = app.state.round_runtime_config
    service = app.state.round_lifecycle_service

    assert config.payload_adapter_kind == TEST_FAMILY_NAME
    assert config.update_family_name == TEST_UPDATE_FAMILY_NAME
    assert config.aggregation_backend_name == TEST_BACKEND_NAME
    assert config.aggregation_backend_overrides == {"max_scale": 1.1}
    assert isinstance(service.round_manager_service.adapter_family, _TestRoundFamily)
