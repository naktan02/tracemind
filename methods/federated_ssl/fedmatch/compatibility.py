"""FedMatch method-local capability compatibility rules."""

from __future__ import annotations

from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.capability_axes import LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT
from methods.federated_ssl.capability_plan import FederatedSslCapabilityPlan
from methods.federated_ssl.fedmatch.descriptor import FEDMATCH_METHOD_NAME


def validate_method_capability_compatibility(
    *,
    method_descriptor: FederatedSslMethodDescriptor,
    capability_plan: FederatedSslCapabilityPlan,
) -> None:
    """FedMatch 전용 local SSL policy는 FedMatch descriptor에서만 허용한다."""

    if capability_plan.local_ssl_policy_name != LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT:
        return
    if method_descriptor.name != FEDMATCH_METHOD_NAME:
        raise ValueError(
            "local_ssl_policy=fedmatch_agreement requires the FedMatch "
            "method-owned descriptor."
        )
