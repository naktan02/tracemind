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
from methods.federated_ssl.capability_axes import (
    LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT,
    LOCAL_SSL_POLICY_FIXMATCH,
    SERVER_UPDATE_FEDAVG_MERGED_DELTA,
    SERVER_UPDATE_FEDMATCH_PARTITIONED,
)
from methods.federated_ssl.capability_plan import (
    LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED,
    LOCAL_SUPERVISION_CLIENT_UNLABELED_ONLY,
    PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN,
    PEER_CONTEXT_NONE,
    QUERY_MULTIVIEW_SOURCE_MATERIALIZED_ROWS,
    SERVER_STEP_NONE,
    SERVER_STEP_SUPERVISED_SEED,
    UPDATE_PARTITION_PARTITIONED,
)
from methods.federated_ssl.fedmatch.original_spec import (
    FEDMATCH_ORIGINAL_COMMIT,
    FEDMATCH_ORIGINAL_REPOSITORY,
)
from shared.src.contracts.common_types import TrainingTaskType

FEDMATCH_METHOD_NAME = "fedmatch"

descriptor = FederatedSslMethodDescriptor(
    name=FEDMATCH_METHOD_NAME,
    display_name="FedMatch",
    implementation_status="partitioned_trainable_state_slice_v1",
    method_role="method_owned",
    required_views=FederatedSslRequiredViews(
        view_names=("text", "aug_0", "aug_1"),
        view_generator_name="usb_multiview",
    ),
    local_step=FederatedSslLocalStepSpec(
        step_name=TrainingTaskType.FEDERATED_SSL_METHOD_LOCAL_STEP.value,
        client_trainer_name="method_owned_local_objective",
        pseudo_labeler_name="fedmatch_agreement_pseudo_labeler",
        training_row_source=TRAINING_ROW_SOURCE_UNLABELED_POOL_WHEN_AVAILABLE,
        runtime_entrypoint=(
            "methods.federated_ssl.fedmatch.partitioned_local_training:"
            "run_method_owned_peft_encoder_training_core"
        ),
    ),
    server_step=FederatedSslServerStepSpec(
        server_aggregator_name="round_runtime_aggregation_backend",
        round_policy_name="round_active_pair_only",
        server_aggregate_hint="fedmatch_peft_text_encoder_partitioned_delta",
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
        requires_custom_server_runtime=False,
    ),
    recipe=FederatedSslMethodRecipe(
        method_name=FEDMATCH_METHOD_NAME,
        supported_local_update_profile_names=("peft_pseudo_label_v1",),
        supported_runtime_pairs=(
            FederatedSslRuntimePair(
                update_family_name="peft_text_encoder",
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
        server_update_policy_names=(
            SERVER_UPDATE_FEDAVG_MERGED_DELTA,
            SERVER_UPDATE_FEDMATCH_PARTITIONED,
        ),
        peer_context_policy_names=(
            PEER_CONTEXT_NONE,
            PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN,
        ),
        update_partition_policy_names=(UPDATE_PARTITION_PARTITIONED,),
        local_ssl_policy_names=(
            LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT,
            LOCAL_SSL_POLICY_FIXMATCH,
        ),
        aggregation_weight_policy_names=(AGGREGATION_WEIGHT_UNIFORM,),
        query_multiview_source_names=(QUERY_MULTIVIEW_SOURCE_MATERIALIZED_ROWS,),
        client_participation_policy_names=(
            PARTICIPATION_ALL_CLIENTS,
            PARTICIPATION_FRACTION_RANDOM,
        ),
    ),
)

ORIGINAL_SOURCE_METADATA = {
    "repository": FEDMATCH_ORIGINAL_REPOSITORY,
    "commit": FEDMATCH_ORIGINAL_COMMIT,
    "paper": "arXiv:2006.12097",
}

DEFAULT_LOCAL_SSL_POLICY_NAME = LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT
DEFAULT_SERVER_UPDATE_POLICY_NAME = SERVER_UPDATE_FEDMATCH_PARTITIONED

TRACE_MAPPING_METADATA = {
    "parameter_decomposition": "peft_text_encoder_sigma_psi",
    "update_partition_policy": UPDATE_PARTITION_PARTITIONED,
    "partition_scheme": "sigma_psi",
    "original_trainable_scope": "ResNet9 Conv/Dense full weights",
    "trace_trainable_scope": "LoRA adapter tensors plus classifier head tensors",
    "frozen_scope": "Transformer backbone base weights",
    "supervised_partition": "sigma",
    "unsupervised_partition": "psi",
    "published_state": "sigma_plus_psi",
    "local_ssl_policy": LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT,
    "current_server_update_policy": SERVER_UPDATE_FEDAVG_MERGED_DELTA,
    "target_server_update_policy": SERVER_UPDATE_FEDMATCH_PARTITIONED,
    "aggregation_weight_policy": AGGREGATION_WEIGHT_UNIFORM,
}

REPORT_TAGS = ("fl_ssl_main_comparison", FEDMATCH_METHOD_NAME)

NOTES = (
    "FedMatch 원본 snapshot의 설정과 core decision semantics를 보존한다.",
    "Method-owned PEFT text encoder local runtime runs FedMatch "
    "supervised/unsupervised partition steps; helper client context selection and "
    "helper weak-view probabilities are wired through the common peer_context axis "
    "and PEFT text encoder update-family slice. The labels-at-server supervised "
    "seed step is wired through the common server_step axis. Sparse S2C/C2S is "
    "wired in the simulation slice with artifact communication estimates, not "
    "measured network packets.",
)
