"""FedMatch round policy metadata."""

from __future__ import annotations

from methods.federated_ssl.peer_context import FederatedSslPeerContextPolicy

FEDMATCH_HELPER_POLICY = "fedmatch_prediction_similarity_topk"

helper_context_policy = FederatedSslPeerContextPolicy(
    policy_name=FEDMATCH_HELPER_POLICY,
    parameters={
        "peer_context_policy": "prediction_similarity_topk",
        "num_helpers": 2,
        "refresh_interval": 10,
    },
)
