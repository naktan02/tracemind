"""server-owned round runtime 조립기."""

from __future__ import annotations

from typing import TYPE_CHECKING

from main_server.src.infrastructure.repositories import (
    shared_adapter_state_repository as shared_adapter_state_repository_module,
)
from main_server.src.services.federation.assets.prototypes import (
    StoredReferencePrototypeRebuildService,
)
from main_server.src.services.federation.rounds.acceptance.models import (
    RoundUpdateAcceptancePolicy,
)
from main_server.src.services.federation.rounds.acceptance.policies import (
    StrictRoundUpdateAcceptancePolicy,
)
from main_server.src.services.federation.rounds.families.registry import (
    build_shared_adapter_round_family,
)
from main_server.src.services.federation.rounds.round_lifecycle_service import (
    RoundLifecycleService,
)
from main_server.src.services.federation.rounds.round_manager_service import (
    RoundManagerService,
)
from main_server.src.services.federation.rounds.runtime_compatibility import (
    validate_server_round_runtime_config,
)
from main_server.src.services.federation.rounds.runtime_config import (
    ServerRoundRuntimeConfig,
)
from shared.src.domain.services.clock import Clock, SystemUtcClock

if TYPE_CHECKING:
    from main_server.src.infrastructure.repositories.round_repository import (
        RoundRepository,
    )


def build_round_manager_service_from_config(
    config: ServerRoundRuntimeConfig,
    *,
    artifact_repository: (
        shared_adapter_state_repository_module.SharedAdapterStateRepository | None
    ) = None,
    clock: Clock | None = None,
) -> RoundManagerService:
    """server-owned runtime config로 RoundManagerService를 조립한다."""

    effective_clock = clock or SystemUtcClock()
    validate_server_round_runtime_config(config)
    adapter_family = build_shared_adapter_round_family(
        config.adapter_family_name,
        aggregation_backend_name=config.aggregation_backend_name,
        aggregation_backend_overrides=config.aggregation_backend_overrides,
    )
    return RoundManagerService(
        adapter_family=adapter_family,
        artifact_repository=(
            artifact_repository
            or shared_adapter_state_repository_module.SharedAdapterStateRepository()
        ),
        clock=effective_clock,
    )


def build_round_lifecycle_service_from_config(
    config: ServerRoundRuntimeConfig,
    *,
    round_repository: RoundRepository | None = None,
    artifact_repository: (
        shared_adapter_state_repository_module.SharedAdapterStateRepository | None
    ) = None,
    prototype_rebuild_runtime_service: (
        StoredReferencePrototypeRebuildService | None
    ) = None,
    update_acceptance_policy: RoundUpdateAcceptancePolicy | None = None,
    clock: Clock | None = None,
) -> RoundLifecycleService:
    """server-owned runtime config로 RoundLifecycleService를 조립한다."""

    from main_server.src.infrastructure.repositories.round_repository import (
        RoundRepository,
    )

    effective_clock = clock or SystemUtcClock()
    return RoundLifecycleService(
        round_repository=round_repository or RoundRepository(),
        round_manager_service=build_round_manager_service_from_config(
            config,
            artifact_repository=artifact_repository,
            clock=effective_clock,
        ),
        prototype_rebuild_runtime_service=prototype_rebuild_runtime_service,
        update_acceptance_policy=(
            update_acceptance_policy or StrictRoundUpdateAcceptancePolicy()
        ),
        clock=effective_clock,
    )
