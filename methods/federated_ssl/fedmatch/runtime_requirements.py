"""FedMatch local runtime requirements."""

from __future__ import annotations

from methods.federated_ssl.capability_axes import LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT


def requires_helper_probability_provider(*, local_ssl_policy_name: str) -> bool:
    """FedMatch agreement objective는 helper weak-view probability를 사용한다."""

    normalized_policy = local_ssl_policy_name.strip().lower().replace("-", "_")
    return normalized_policy == LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT
