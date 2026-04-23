# ruff: noqa: F401

"""Main-server federated round services."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .acceptance.errors import (
    RoundConflictError,
    RoundValidationError,
)
from .acceptance.models import (
    RoundNetworkPolicy,
    RoundTrustPolicy,
    RoundUpdateAcceptancePolicy,
)
from .acceptance.network_policies import (
    IdempotentRoundNetworkPolicy,
    StrictRoundNetworkPolicy,
)
from .acceptance.policies import (
    CompositeRoundUpdateAcceptancePolicy,
    IdempotentRoundUpdateAcceptancePolicy,
    StrictRoundUpdateAcceptancePolicy,
)
from .acceptance.trust_policies import (
    AllowAllRoundTrustPolicy,
    SingleSubmissionPerAgentTrustPolicy,
)
from .aggregation.classifier_head import ClassifierHeadFedAvgAggregationService
from .aggregation.diagonal_scale import DiagonalScaleAggregationService
from .aggregation.diagonal_scale_defaults import (
    DEFAULT_DIAGONAL_SCALE_FEDAVG_AGGREGATION_CONFIG,
    AggregationConfigScalar,
    DiagonalScaleFedAvgAggregationConfig,
)
from .aggregation.models import (
    AggregationConfig,
    AggregationResult,
    SharedAdapterAggregationBackend,
)
from .aggregation.registry import (
    build_shared_adapter_aggregation_backend,
    list_registered_shared_adapter_aggregation_backends,
    list_shared_adapter_aggregation_backend_catalog_entries,
    register_shared_adapter_aggregation_backend,
)
from .families.classifier_head import ClassifierHeadRoundFamily
from .families.diagonal_scale import DiagonalScaleRoundFamily
from .families.models import SharedAdapterRoundFamily
from .families.registry import (
    build_shared_adapter_round_family,
    register_shared_adapter_round_family,
)
from .round_lifecycle_service import RoundLifecycleService
from .round_manager_service import (
    RoundManagerService,
    RoundPublication,
)
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

AggregationService = DiagonalScaleAggregationService

if TYPE_CHECKING:
    from main_server.src.infrastructure.repositories import (
        shared_adapter_state_repository as shared_adapter_state_repository_module,
    )
    from main_server.src.infrastructure.repositories.round_repository import (
        RoundRepository,
    )
    from main_server.src.services.federation.assets.prototypes import (
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
