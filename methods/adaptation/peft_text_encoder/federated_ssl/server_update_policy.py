"""PEFT text encoder/head family의 FL SSL server update policy 해석."""

from __future__ import annotations

from methods.adaptation.federated_ssl_server_update import (
    register_federated_ssl_server_update_backend_resolver,
)
from methods.adaptation.peft_text_encoder.aggregation import (
    peft_encoder_partitioned_projection as peft_part_projection,
)
from methods.federated_ssl.capability_axes import SERVER_UPDATE_FEDMATCH_PARTITIONED
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_ADAPTER_KIND,
)


@register_federated_ssl_server_update_backend_resolver(PEFT_CLASSIFIER_ADAPTER_KIND)
def resolve_peft_encoder_federated_ssl_server_update_backend(
    *,
    server_update_policy_name: str,
    aggregation_backend_name: str,
) -> str:
    """PEFT text encoder/head server update policy를 aggregation backend로 해석한다."""

    if server_update_policy_name != SERVER_UPDATE_FEDMATCH_PARTITIONED:
        raise ValueError(
            "Unsupported PEFT text encoder/head FL SSL server_update_policy: "
            f"{server_update_policy_name}."
        )
    normalized_backend = aggregation_backend_name.strip().lower()
    if normalized_backend != "fedavg":
        raise ValueError(
            "server_update_policy=fedmatch_partitioned currently maps from "
            "round_runtime.aggregation_backend_name=fedavg to "
            f"{peft_part_projection.PARTITIONED_DELTA_AVERAGE_BACKEND_NAME}."
        )
    return peft_part_projection.PARTITIONED_DELTA_AVERAGE_BACKEND_NAME
