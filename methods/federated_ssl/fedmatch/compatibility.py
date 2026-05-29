"""FedMatch method-local capability compatibility rules."""

from __future__ import annotations

from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.capability_axes import (
    LOCAL_SSL_POLICIES_REQUIRING_STATE_SURFACE,
    LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT,
    LOCAL_SSL_POLICY_FIXMATCH,
    SERVER_UPDATE_FEDMATCH_PARTITIONED,
)
from methods.federated_ssl.capability_plan import (
    UPDATE_PARTITION_PARTITIONED,
    FederatedSslCapabilityPlan,
)
from methods.federated_ssl.fedmatch.descriptor import FEDMATCH_METHOD_NAME
from methods.federated_ssl.method_module_resolution import (
    resolve_federated_ssl_method_family_name,
)

_FEDMATCH_PARTITIONED_SIMULATION_LOCAL_SSL_POLICIES = frozenset(
    {
        LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT,
        LOCAL_SSL_POLICY_FIXMATCH,
    }
)


def validate_method_capability_compatibility(
    *,
    method_descriptor: FederatedSslMethodDescriptor,
    capability_plan: FederatedSslCapabilityPlan,
) -> None:
    """FedMatch 전용 local SSL policy는 FedMatch descriptor에서만 허용한다."""

    _validate_partitioned_server_update_policy(capability_plan)
    if capability_plan.local_ssl_policy_name != LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT:
        return
    if (
        resolve_federated_ssl_method_family_name(method_descriptor.name)
        != FEDMATCH_METHOD_NAME
    ):
        raise ValueError(
            "local_ssl_policy=fedmatch_agreement requires the FedMatch "
            "method-owned descriptor."
        )
    if capability_plan.server_update_policy_name != SERVER_UPDATE_FEDMATCH_PARTITIONED:
        raise ValueError(
            "local_ssl_policy=fedmatch_agreement requires "
            "server_update_policy=fedmatch_partitioned so sigma/psi partitioned "
            "state is preserved across rounds."
        )


def validate_method_simulation_runtime_support(
    *,
    method_descriptor: FederatedSslMethodDescriptor,
    capability_plan: FederatedSslCapabilityPlan,
) -> None:
    """FedMatch simulation runtime이 현재 생산 가능한 policy 조합인지 검증한다."""

    del method_descriptor
    if capability_plan.server_update_policy_name != SERVER_UPDATE_FEDMATCH_PARTITIONED:
        return
    if capability_plan.local_ssl_policy_name not in (
        _FEDMATCH_PARTITIONED_SIMULATION_LOCAL_SSL_POLICIES
    ):
        raise ValueError(
            "server_update_policy=fedmatch_partitioned currently requires "
            "local_ssl_policy=fedmatch_agreement or fixmatch in simulation. "
            "Stateful Query SSL local objectives with partitioned sigma/psi loops "
            "need a state surface first."
        )


def _validate_partitioned_server_update_policy(
    capability_plan: FederatedSslCapabilityPlan,
) -> None:
    if capability_plan.server_update_policy_name != SERVER_UPDATE_FEDMATCH_PARTITIONED:
        return
    if capability_plan.update_partition_policy_name != UPDATE_PARTITION_PARTITIONED:
        raise ValueError(
            "server_update_policy=fedmatch_partitioned requires "
            "update_partition_policy=partitioned."
        )
    if (
        capability_plan.local_ssl_policy_name
        in LOCAL_SSL_POLICIES_REQUIRING_STATE_SURFACE
    ):
        raise ValueError(
            "server_update_policy=fedmatch_partitioned with "
            f"local_ssl_policy={capability_plan.local_ssl_policy_name} requires a "
            "local SSL state surface before execution."
        )
