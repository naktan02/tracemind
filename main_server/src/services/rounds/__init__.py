"""Main-server federated round services."""

from .adapter_family_service import (
    DiagonalScaleRoundFamily,
    SharedAdapterRoundFamily,
    build_shared_adapter_round_family,
    register_shared_adapter_round_family,
)
from .aggregation_service import (
    AggregationConfig,
    AggregationResult,
    AggregationService,
    DiagonalScaleAggregationService,
    SharedAdapterAggregationBackend,
    build_shared_adapter_aggregation_backend,
    register_shared_adapter_aggregation_backend,
)
from .round_manager_service import RoundManagerService, RoundPublication
from .runtime_config import (
    ROUND_ADAPTER_FAMILY_ENV,
    ROUND_AGGREGATION_BACKEND_ENV,
    ServerRoundRuntimeConfig,
    load_server_round_runtime_config_from_env,
)
from .runtime_factory import (
    build_round_lifecycle_service_from_config,
    build_round_manager_service_from_config,
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

__all__ = [
    "AggregationConfig",
    "AggregationResult",
    "AggregationService",
    "AllowAllRoundTrustPolicy",
    "build_round_lifecycle_service_from_config",
    "build_round_manager_service_from_config",
    "build_shared_adapter_aggregation_backend",
    "build_shared_adapter_round_family",
    "CompositeRoundUpdateAcceptancePolicy",
    "DiagonalScaleRoundFamily",
    "DiagonalScaleAggregationService",
    "IdempotentRoundUpdateAcceptancePolicy",
    "IdempotentRoundNetworkPolicy",
    "register_shared_adapter_aggregation_backend",
    "register_shared_adapter_round_family",
    "RoundConflictError",
    "RoundNetworkPolicy",
    "ROUND_ADAPTER_FAMILY_ENV",
    "ROUND_AGGREGATION_BACKEND_ENV",
    "ServerRoundRuntimeConfig",
    "SharedAdapterRoundFamily",
    "SharedAdapterAggregationBackend",
    "load_server_round_runtime_config_from_env",
    "RoundManagerService",
    "RoundPublication",
    "RoundTrustPolicy",
    "RoundUpdateAcceptancePolicy",
    "RoundValidationError",
    "SingleSubmissionPerAgentTrustPolicy",
    "StrictRoundNetworkPolicy",
    "StrictRoundUpdateAcceptancePolicy",
]
