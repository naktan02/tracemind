"""Main-server federated round services."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .adapter_family_service import (
    ClassifierHeadRoundFamily,
    DiagonalScaleRoundFamily,
    SharedAdapterRoundFamily,
    build_shared_adapter_round_family,
    register_shared_adapter_round_family,
)
from .aggregation_service import (
    AggregationConfig,
    AggregationResult,
    AggregationService,
    ClassifierHeadFedAvgAggregationService,
    DiagonalScaleAggregationService,
    SharedAdapterAggregationBackend,
    build_shared_adapter_aggregation_backend,
    register_shared_adapter_aggregation_backend,
)
from .round_lifecycle_service import RoundLifecycleService
from .round_manager_service import RoundManagerService, RoundPublication
from .runtime_compatibility import (
    ServerRoundRuntimeCompatibility,
    validate_server_round_runtime_config,
)
from .runtime_config import (
    ROUND_ADAPTER_FAMILY_ENV,
    ROUND_AGGREGATION_BACKEND_CONFIG_ENV,
    ROUND_AGGREGATION_BACKEND_ENV,
    ServerRoundRuntimeConfig,
    load_server_round_runtime_config_from_env,
)
from .update_acceptance_policy import (
    AllowAllRoundTrustPolicy,
    CompositeRoundUpdateAcceptancePolicy,
    IdempotentRoundNetworkPolicy,
    IdempotentRoundUpdateAcceptancePolicy,
    RoundConflictError,
    RoundNetworkPolicy,
    RoundTrustPolicy,
    RoundUpdateAcceptancePolicy,
    RoundValidationError,
    SingleSubmissionPerAgentTrustPolicy,
    StrictRoundNetworkPolicy,
    StrictRoundUpdateAcceptancePolicy,
)

if TYPE_CHECKING:
    from main_server.src.infrastructure.repositories import (
        shared_adapter_state_repository as shared_adapter_state_repository_module,
    )
    from main_server.src.infrastructure.repositories.round_repository import (
        RoundRepository,
    )
    from main_server.src.services.prototypes.prototype_rebuild_service import (
        StoredReferencePrototypeRebuildService,
    )
    from shared.src.domain.services.clock import Clock


def build_round_manager_service_from_config(
    config: ServerRoundRuntimeConfig,
    *,
    artifact_repository: (
        shared_adapter_state_repository_module.SharedAdapterStateRepository | None
    ) = None,
    clock: Clock | None = None,
) -> RoundManagerService:
    from .runtime_factory import build_round_manager_service_from_config as _build

    return _build(
        config,
        artifact_repository=artifact_repository,
        clock=clock,
    )


def build_round_lifecycle_service_from_config(
    config: ServerRoundRuntimeConfig,
    *,
    round_repository: RoundRepository | None = None,
    artifact_repository: (
        shared_adapter_state_repository_module.SharedAdapterStateRepository | None
    ) = None,
    prototype_rebuild_runtime_service: StoredReferencePrototypeRebuildService
    | None = None,
    update_acceptance_policy: RoundUpdateAcceptancePolicy | None = None,
    clock: Clock | None = None,
) -> RoundLifecycleService:
    from .runtime_factory import build_round_lifecycle_service_from_config as _build

    return _build(
        config,
        round_repository=round_repository,
        artifact_repository=artifact_repository,
        prototype_rebuild_runtime_service=prototype_rebuild_runtime_service,
        update_acceptance_policy=update_acceptance_policy,
        clock=clock,
    )

__all__ = [
    "AggregationConfig",
    "AggregationResult",
    "AggregationService",
    "AllowAllRoundTrustPolicy",
    "build_round_lifecycle_service_from_config",
    "build_round_manager_service_from_config",
    "build_shared_adapter_aggregation_backend",
    "build_shared_adapter_round_family",
    "ClassifierHeadFedAvgAggregationService",
    "ClassifierHeadRoundFamily",
    "CompositeRoundUpdateAcceptancePolicy",
    "DiagonalScaleRoundFamily",
    "DiagonalScaleAggregationService",
    "IdempotentRoundUpdateAcceptancePolicy",
    "IdempotentRoundNetworkPolicy",
    "register_shared_adapter_aggregation_backend",
    "register_shared_adapter_round_family",
    "RoundConflictError",
    "RoundLifecycleService",
    "RoundNetworkPolicy",
    "ROUND_ADAPTER_FAMILY_ENV",
    "ROUND_AGGREGATION_BACKEND_ENV",
    "ROUND_AGGREGATION_BACKEND_CONFIG_ENV",
    "ServerRoundRuntimeConfig",
    "SharedAdapterRoundFamily",
    "SharedAdapterAggregationBackend",
    "load_server_round_runtime_config_from_env",
    "RoundManagerService",
    "RoundPublication",
    "ServerRoundRuntimeCompatibility",
    "RoundTrustPolicy",
    "RoundUpdateAcceptancePolicy",
    "RoundValidationError",
    "SingleSubmissionPerAgentTrustPolicy",
    "StrictRoundNetworkPolicy",
    "StrictRoundUpdateAcceptancePolicy",
    "validate_server_round_runtime_config",
]
