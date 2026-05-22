"""FedMatch round policy metadata."""

from __future__ import annotations

from methods.federated_ssl.fedmatch.helper_selection import (
    FEDMATCH_DEFAULT_NUM_HELPERS,
    FEDMATCH_DEFAULT_REFRESH_INTERVAL,
    FEDMATCH_HELPER_SELECTION_NAME,
)
from methods.federated_ssl.peer_context import FederatedSslPeerContextPolicy

FEDMATCH_HELPER_POLICY = "fedmatch_prediction_similarity_topk"

helper_context_policy = FederatedSslPeerContextPolicy(
    policy_name=FEDMATCH_HELPER_POLICY,
    parameters={
        "peer_context_policy": FEDMATCH_HELPER_SELECTION_NAME,
        "num_helpers": FEDMATCH_DEFAULT_NUM_HELPERS,
        "refresh_interval": FEDMATCH_DEFAULT_REFRESH_INTERVAL,
        "selection_metric": "model_output_vector_distance",
    },
)
