"""FedMatch labels-at-client canonical variant descriptor."""

from __future__ import annotations

from methods.federated.aggregation_weighting import AGGREGATION_WEIGHT_UNIFORM
from methods.federated.client_split import LABELED_EXPOSURE_SHARED_CLIENT_SEED
from methods.federated.participation import (
    PARTICIPATION_ALL_CLIENTS,
    PARTICIPATION_FRACTION_RANDOM,
)
from methods.federated_ssl.base import (
    FederatedSslMethodDescriptor,
    FederatedSslMethodRecipe,
    FederatedSslRequiredCapabilities,
)
from methods.federated_ssl.capability_axes import (
    LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT,
    SERVER_UPDATE_FEDMATCH_PARTITIONED,
)
from methods.federated_ssl.capability_plan import (
    LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED,
    PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN,
    QUERY_MULTIVIEW_SOURCE_MATERIALIZED_ROWS,
    SERVER_STEP_NONE,
    UPDATE_PARTITION_PARTITIONED,
)
from methods.federated_ssl.fedmatch.descriptor import (
    NOTES as FEDMATCH_NOTES,
)
from methods.federated_ssl.fedmatch.descriptor import (
    ORIGINAL_SOURCE_METADATA as FEDMATCH_ORIGINAL_SOURCE_METADATA,
)
from methods.federated_ssl.fedmatch.descriptor import (
    TRACE_MAPPING_METADATA as FEDMATCH_TRACE_MAPPING_METADATA,
)
from methods.federated_ssl.fedmatch.descriptor import (
    descriptor as fedmatch_descriptor,
)
from methods.federated_ssl.fedmatch.original_spec import (
    FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
)

METHOD_IMPLEMENTATION_FAMILY = "fedmatch"
FEDMATCH_LABELS_AT_CLIENT_METHOD_NAME = "fedmatch_labels_at_client"

descriptor = FederatedSslMethodDescriptor(
    name=FEDMATCH_LABELS_AT_CLIENT_METHOD_NAME,
    display_name="FedMatch (labels-at-client)",
    implementation_status=fedmatch_descriptor.implementation_status,
    method_role=fedmatch_descriptor.method_role,
    required_views=fedmatch_descriptor.required_views,
    local_step=fedmatch_descriptor.local_step,
    server_step=fedmatch_descriptor.server_step,
    round_state_exchange=fedmatch_descriptor.round_state_exchange,
    runtime_capabilities=fedmatch_descriptor.runtime_capabilities,
    recipe=FederatedSslMethodRecipe(
        method_name=FEDMATCH_LABELS_AT_CLIENT_METHOD_NAME,
        supported_local_update_profile_names=(
            fedmatch_descriptor.recipe.supported_local_update_profile_names
        ),
        supported_runtime_pairs=fedmatch_descriptor.recipe.supported_runtime_pairs,
    ),
    required_capabilities=FederatedSslRequiredCapabilities(
        labeled_exposure_policy_names=(LABELED_EXPOSURE_SHARED_CLIENT_SEED,),
        local_supervision_regime_names=(
            LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED,
        ),
        server_step_policy_names=(SERVER_STEP_NONE,),
        server_update_policy_names=(SERVER_UPDATE_FEDMATCH_PARTITIONED,),
        peer_context_policy_names=(PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN,),
        update_partition_policy_names=(UPDATE_PARTITION_PARTITIONED,),
        local_ssl_policy_names=(LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT,),
        aggregation_weight_policy_names=(AGGREGATION_WEIGHT_UNIFORM,),
        query_multiview_source_names=(QUERY_MULTIVIEW_SOURCE_MATERIALIZED_ROWS,),
        client_participation_policy_names=(
            PARTICIPATION_ALL_CLIENTS,
            PARTICIPATION_FRACTION_RANDOM,
        ),
    ),
)

DEFAULT_LOCAL_SSL_POLICY_NAME = LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT
DEFAULT_SERVER_UPDATE_POLICY_NAME = SERVER_UPDATE_FEDMATCH_PARTITIONED
DEFAULT_SERVER_STEP_POLICY_NAME = SERVER_STEP_NONE
DEFAULT_PEER_CONTEXT_POLICY_NAME = PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN

ORIGINAL_SOURCE_METADATA = dict(FEDMATCH_ORIGINAL_SOURCE_METADATA)
TRACE_MAPPING_METADATA = {
    **FEDMATCH_TRACE_MAPPING_METADATA,
    "scenario": FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
    "server_step_policy": SERVER_STEP_NONE,
    "peer_context_policy": PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN,
}
REPORT_TAGS = (
    "fl_ssl_main_comparison",
    "fedmatch",
    FEDMATCH_LABELS_AT_CLIENT_METHOD_NAME,
)
NOTES = (
    *FEDMATCH_NOTES,
    "이 variant는 labels-at-client canonical FedMatch 조합을 method-owned "
    "descriptor 하나로 닫는다.",
)
