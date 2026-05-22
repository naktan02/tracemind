"""FedMatch FL SSL method descriptor."""

from __future__ import annotations

from methods.federated.aggregation_weighting import AGGREGATION_WEIGHT_UNIFORM
from methods.federated.client_split import (
    LABELED_EXPOSURE_SERVER_ONLY_SEED,
    LABELED_EXPOSURE_SHARED_CLIENT_SEED,
)
from methods.federated.participation import (
    PARTICIPATION_ALL_CLIENTS,
    PARTICIPATION_FRACTION_RANDOM,
)
from methods.federated_ssl.base import (
    TRAINING_ROW_SOURCE_UNLABELED_POOL_WHEN_AVAILABLE,
    FederatedSslLocalStepSpec,
    FederatedSslMethodDescriptor,
    FederatedSslMethodRecipe,
    FederatedSslRequiredCapabilities,
    FederatedSslRequiredViews,
    FederatedSslRoundStateExchangeSpec,
    FederatedSslRuntimeCapabilities,
    FederatedSslRuntimePair,
    FederatedSslServerStepSpec,
)
from methods.federated_ssl.capability_plan import (
    LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED,
    LOCAL_SUPERVISION_CLIENT_UNLABELED_ONLY,
    PEER_CONTEXT_NONE,
    PEER_CONTEXT_PREDICTION_SIMILARITY_TOPK,
    QUERY_MULTIVIEW_SOURCE_MATERIALIZED_ROWS,
    SERVER_STEP_NONE,
    SERVER_STEP_SUPERVISED_SEED,
    UPDATE_PARTITION_SIGMA_PSI,
)

FEDMATCH_METHOD_NAME = "fedmatch"

descriptor = FederatedSslMethodDescriptor(
    name=FEDMATCH_METHOD_NAME,
    implementation_status="capability_surface_v1",
    method_role="method_owned",
    required_views=FederatedSslRequiredViews(
        view_names=("text", "aug_0", "aug_1"),
        view_generator_name="usb_multiview",
    ),
    local_step=FederatedSslLocalStepSpec(
        step_name="fedmatch_local_step",
        client_trainer_name="method_owned_local_objective",
        pseudo_labeler_name="fedmatch_agreement_pseudo_labeler",
        training_row_source=TRAINING_ROW_SOURCE_UNLABELED_POOL_WHEN_AVAILABLE,
    ),
    server_step=FederatedSslServerStepSpec(
        server_aggregator_name="round_runtime_aggregation_backend",
        round_policy_name="method_owned_server_step",
        server_aggregate_hint="fedmatch_sigma_psi_partitioned_update",
    ),
    round_state_exchange=FederatedSslRoundStateExchangeSpec(
        exchange_name="peer_context",
        required_client_metric_keys=("mean_confidence",),
        summary_metric_prefix="fedmatch",
        requires_custom_exchange=True,
    ),
    runtime_capabilities=FederatedSslRuntimeCapabilities(
        simulation_supported=True,
        live_agent_supported=False,
        live_server_supported=False,
        requires_custom_client_runtime=True,
        requires_custom_server_runtime=True,
    ),
    recipe=FederatedSslMethodRecipe(
        method_name=FEDMATCH_METHOD_NAME,
        supported_local_update_profile_names=("lora_pseudo_label_v1",),
        supported_runtime_pairs=(
            FederatedSslRuntimePair(
                adapter_family_name="lora_classifier",
                aggregation_backend_name="fedavg",
            ),
        ),
    ),
    required_capabilities=FederatedSslRequiredCapabilities(
        labeled_exposure_policy_names=(
            LABELED_EXPOSURE_SHARED_CLIENT_SEED,
            LABELED_EXPOSURE_SERVER_ONLY_SEED,
        ),
        local_supervision_regime_names=(
            LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED,
            LOCAL_SUPERVISION_CLIENT_UNLABELED_ONLY,
        ),
        server_step_policy_names=(SERVER_STEP_NONE, SERVER_STEP_SUPERVISED_SEED),
        peer_context_policy_names=(
            PEER_CONTEXT_NONE,
            PEER_CONTEXT_PREDICTION_SIMILARITY_TOPK,
        ),
        update_partition_policy_names=(UPDATE_PARTITION_SIGMA_PSI,),
        aggregation_weight_policy_names=(AGGREGATION_WEIGHT_UNIFORM,),
        query_multiview_source_names=(QUERY_MULTIVIEW_SOURCE_MATERIALIZED_ROWS,),
        client_participation_policy_names=(
            PARTICIPATION_ALL_CLIENTS,
            PARTICIPATION_FRACTION_RANDOM,
        ),
    ),
)
