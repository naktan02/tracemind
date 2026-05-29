"""FedMatch server-side step parameter resolution."""

from __future__ import annotations

from collections.abc import Mapping

from methods.federated_ssl.server_step import (
    FederatedSslSupervisedSeedStepParameters,
)


def resolve_supervised_seed_step_parameters(
    *,
    effective_parameters: Mapping[str, object] | None,
    default_epochs: int,
    default_batch_size: int,
    round_index: int,
) -> FederatedSslSupervisedSeedStepParameters:
    """FedMatch 원본 parameter 이름에서 server seed step budget을 해석한다."""

    parameters = {} if effective_parameters is None else dict(effective_parameters)
    epoch_key = "server_pretrain_epochs" if round_index == 1 else "server_epochs"
    return FederatedSslSupervisedSeedStepParameters(
        epochs=int(parameters.get(epoch_key, default_epochs)),
        batch_size=int(parameters.get("server_batch_size", default_batch_size)),
    )
