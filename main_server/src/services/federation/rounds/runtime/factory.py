"""server-owned round runtime 조립기."""

from __future__ import annotations

from typing import TYPE_CHECKING

from main_server.src.infrastructure.repositories import (
    model_manifest_repository as model_manifest_repository_module,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_state_repository as shared_adapter_state_repository_module,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_update_repository as shared_adapter_update_repository_module,
)
from main_server.src.services.federation.rounds.acceptance.models import (
    RoundUpdateAcceptancePolicy,
)
from main_server.src.services.federation.rounds.acceptance.policies import (
    StrictRoundUpdateAcceptancePolicy,
)
from main_server.src.services.federation.rounds.active_manifest_service import (
    ActiveModelManifestService,
)
from main_server.src.services.federation.rounds.payload_adapters.registry import (
    build_shared_adapter_round_payload_adapter,
)
from main_server.src.services.federation.rounds.round_lifecycle_service import (
    RoundLifecycleService,
)
from main_server.src.services.federation.rounds.round_manager_service import (
    RoundManagerService,
)
from main_server.src.services.federation.rounds.runtime.compatibility import (
    validate_server_round_runtime_config,
)
from main_server.src.services.federation.rounds.runtime.config import (
    ServerRoundRuntimeConfig,
)
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor
from shared.src.domain.services.clock import Clock, SystemUtcClock

if TYPE_CHECKING:
    from main_server.src.infrastructure.repositories.round_repository import (
        RoundRepository,
    )

SharedAdapterUpdateRepository = (
    shared_adapter_update_repository_module.SharedAdapterUpdateRepository
)
ModelManifestRepository = model_manifest_repository_module.ModelManifestRepository


def build_round_manager_service_from_config(
    config: ServerRoundRuntimeConfig,
    *,
    artifact_repository: (
        shared_adapter_state_repository_module.SharedAdapterStateRepository | None
    ) = None,
    update_payload_repository: SharedAdapterUpdateRepository | None = None,
    clock: Clock | None = None,
) -> RoundManagerService:
    """server-owned runtime config로 RoundManagerService를 조립한다."""

    effective_clock = clock or SystemUtcClock()
    validate_server_round_runtime_config(config)
    payload_adapter = build_shared_adapter_round_payload_adapter(
        config.payload_adapter_kind,
        aggregation_backend_name=config.aggregation_backend_name,
        aggregation_backend_overrides=config.aggregation_backend_overrides,
    )
    return RoundManagerService(
        payload_adapter=payload_adapter,
        artifact_repository=(
            artifact_repository
            or shared_adapter_state_repository_module.SharedAdapterStateRepository()
        ),
        update_payload_repository=(
            update_payload_repository or SharedAdapterUpdateRepository()
        ),
        clock=effective_clock,
    )


def build_round_lifecycle_service_from_config(
    config: ServerRoundRuntimeConfig,
    *,
    round_repository: RoundRepository | None = None,
    update_payload_repository: SharedAdapterUpdateRepository | None = None,
    model_manifest_repository: ModelManifestRepository | None = None,
    active_manifest_service: ActiveModelManifestService | None = None,
    artifact_repository: (
        shared_adapter_state_repository_module.SharedAdapterStateRepository | None
    ) = None,
    update_acceptance_policy: RoundUpdateAcceptancePolicy | None = None,
    clock: Clock | None = None,
) -> RoundLifecycleService:
    """server-owned runtime config로 RoundLifecycleService를 조립한다."""

    from main_server.src.infrastructure.repositories.round_repository import (
        RoundRepository,
    )

    effective_clock = clock or SystemUtcClock()
    effective_update_payload_repository = (
        update_payload_repository or SharedAdapterUpdateRepository()
    )
    compatibility = validate_server_round_runtime_config(config)
    method_descriptor = (
        None
        if compatibility.method_descriptor_name is None
        else resolve_federated_ssl_method_descriptor(
            compatibility.method_descriptor_name
        )
    )
    return RoundLifecycleService(
        round_repository=round_repository or RoundRepository(),
        update_payload_repository=effective_update_payload_repository,
        active_manifest_service=(
            active_manifest_service
            or ActiveModelManifestService(
                manifest_repository=(
                    model_manifest_repository or ModelManifestRepository()
                ),
                clock=effective_clock,
            )
        ),
        round_manager_service=build_round_manager_service_from_config(
            config,
            artifact_repository=artifact_repository,
            update_payload_repository=effective_update_payload_repository,
            clock=effective_clock,
        ),
        update_acceptance_policy=(
            update_acceptance_policy or StrictRoundUpdateAcceptancePolicy()
        ),
        method_descriptor=method_descriptor,
        round_runtime_config=config,
        clock=effective_clock,
    )
