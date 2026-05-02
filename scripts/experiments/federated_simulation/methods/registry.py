"""FL SSL method 선택 축의 얇은 descriptor registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FederatedSslMethodDescriptor:
    """실험 method가 어느 runtime 경계를 쓰는지 설명하는 descriptor."""

    name: str
    implementation_status: str
    requires_custom_client_runtime: bool
    requires_custom_server_runtime: bool


_REGISTERED_METHODS = {
    "fedavg_pseudo_label": FederatedSslMethodDescriptor(
        name="fedavg_pseudo_label",
        implementation_status="active_runtime",
        requires_custom_client_runtime=False,
        requires_custom_server_runtime=False,
    )
}


def resolve_federated_ssl_method(name: str) -> FederatedSslMethodDescriptor:
    """method 이름을 baseline descriptor로 해석한다."""
    normalized_name = name.strip().lower()
    descriptor = _REGISTERED_METHODS.get(normalized_name)
    if descriptor is None:
        raise NotImplementedError(
            "Federated SSL method is not wired yet. "
            f"Choose one of {sorted(_REGISTERED_METHODS)} or add the method "
            f"descriptor first: {name}"
        )
    return descriptor
