"""FedMatch method-owned policy, default, report surface."""

from __future__ import annotations

from collections.abc import Mapping

from methods.federated.aggregation_weighting import AGGREGATION_WEIGHT_UNIFORM
from methods.federated.client_split import (
    LABELED_EXPOSURE_SERVER_ONLY_SEED,
    LABELED_EXPOSURE_SHARED_CLIENT_SEED,
)
from methods.federated_ssl.capabilities.axes import (
    LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT,
    SERVER_UPDATE_FEDAVG_MERGED_DELTA,
    SERVER_UPDATE_FEDMATCH_PARTITIONED,
)
from methods.federated_ssl.capabilities.plan import (
    LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED,
    LOCAL_SUPERVISION_CLIENT_UNLABELED_ONLY,
    PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN,
    SERVER_STEP_NONE,
    SERVER_STEP_SUPERVISED_SEED,
    UPDATE_PARTITION_PARTITIONED,
)
from methods.federated_ssl.fedmatch.helper_selection import (
    FEDMATCH_DEFAULT_NUM_HELPERS,
    FEDMATCH_DEFAULT_REFRESH_INTERVAL,
    FEDMATCH_HELPER_SELECTION_NAME,
)
from methods.federated_ssl.fedmatch.original_spec import (
    FEDMATCH_ORIGINAL_COMMIT,
    FEDMATCH_ORIGINAL_REPOSITORY,
    FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
    FEDMATCH_SCENARIO_LABELS_AT_SERVER,
)
from methods.federated_ssl.fedmatch.partitioning import (
    FEDMATCH_PSI_PARTITION,
    FEDMATCH_SIGMA_PARTITION,
)
from methods.federated_ssl.hooks.peer_context import FederatedSslPeerContextPolicy
from methods.federated_ssl.hooks.server_step import (
    FederatedSslServerStepPolicy,
    FederatedSslSupervisedSeedStepParameters,
)

FEDMATCH_METHOD_NAME = "fedmatch"

FEDMATCH_LABELS_AT_CLIENT_POLICY = "fedmatch_labels_at_client"
FEDMATCH_LABELS_AT_SERVER_POLICY = "fedmatch_labels_at_server"
FEDMATCH_HELPER_POLICY = "fedmatch_fixed_probe_output_knn"

ORIGINAL_SOURCE_METADATA = {
    "repository": FEDMATCH_ORIGINAL_REPOSITORY,
    "commit": FEDMATCH_ORIGINAL_COMMIT,
    "paper": "arXiv:2006.12097",
}

DEFAULT_LOCAL_SSL_POLICY_NAME = LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT
DEFAULT_SERVER_UPDATE_POLICY_NAME = SERVER_UPDATE_FEDMATCH_PARTITIONED
DEFAULT_LABELED_EXPOSURE_POLICY_BY_SCENARIO = {
    FEDMATCH_SCENARIO_LABELS_AT_CLIENT: LABELED_EXPOSURE_SHARED_CLIENT_SEED,
    FEDMATCH_SCENARIO_LABELS_AT_SERVER: LABELED_EXPOSURE_SERVER_ONLY_SEED,
}
DEFAULT_LOCAL_SUPERVISION_REGIME_BY_SCENARIO = {
    FEDMATCH_SCENARIO_LABELS_AT_CLIENT: LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED,
    FEDMATCH_SCENARIO_LABELS_AT_SERVER: LOCAL_SUPERVISION_CLIENT_UNLABELED_ONLY,
}
DEFAULT_SERVER_STEP_POLICY_BY_SCENARIO = {
    FEDMATCH_SCENARIO_LABELS_AT_CLIENT: SERVER_STEP_NONE,
    FEDMATCH_SCENARIO_LABELS_AT_SERVER: SERVER_STEP_SUPERVISED_SEED,
}
DEFAULT_PEER_CONTEXT_POLICY_BY_SCENARIO = {
    FEDMATCH_SCENARIO_LABELS_AT_CLIENT: PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN,
    FEDMATCH_SCENARIO_LABELS_AT_SERVER: PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN,
}

TRACE_MAPPING_METADATA = {
    "parameter_decomposition": "peft_text_encoder_sigma_psi",
    "update_partition_policy": UPDATE_PARTITION_PARTITIONED,
    "partition_scheme": "sigma_psi",
    "original_trainable_scope": "ResNet9 Conv/Dense full weights",
    "trace_trainable_scope": "LoRA adapter tensors plus classifier head tensors",
    "frozen_scope": "Transformer backbone base weights",
    "supervised_partition": FEDMATCH_SIGMA_PARTITION,
    "unsupervised_partition": FEDMATCH_PSI_PARTITION,
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

FEDMATCH_METHOD_CONFIG_SURFACE: Mapping[str, object] = {
    "ORIGINAL_SOURCE_METADATA": ORIGINAL_SOURCE_METADATA,
    "DEFAULT_LOCAL_SSL_POLICY_NAME": DEFAULT_LOCAL_SSL_POLICY_NAME,
    "DEFAULT_SERVER_UPDATE_POLICY_NAME": DEFAULT_SERVER_UPDATE_POLICY_NAME,
    "DEFAULT_LABELED_EXPOSURE_POLICY_BY_SCENARIO": (
        DEFAULT_LABELED_EXPOSURE_POLICY_BY_SCENARIO
    ),
    "DEFAULT_LOCAL_SUPERVISION_REGIME_BY_SCENARIO": (
        DEFAULT_LOCAL_SUPERVISION_REGIME_BY_SCENARIO
    ),
    "DEFAULT_SERVER_STEP_POLICY_BY_SCENARIO": DEFAULT_SERVER_STEP_POLICY_BY_SCENARIO,
    "DEFAULT_PEER_CONTEXT_POLICY_BY_SCENARIO": DEFAULT_PEER_CONTEXT_POLICY_BY_SCENARIO,
    "TRACE_MAPPING_METADATA": TRACE_MAPPING_METADATA,
    "REPORT_TAGS": REPORT_TAGS,
    "NOTES": NOTES,
}
FEDMATCH_METHOD_CONFIG_SURFACE_BY_METHOD_NAME = {
    FEDMATCH_METHOD_NAME: FEDMATCH_METHOD_CONFIG_SURFACE,
}

labels_at_client_policy = FederatedSslServerStepPolicy(
    policy_name=FEDMATCH_LABELS_AT_CLIENT_POLICY,
    parameters={
        "scenario": FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
        "server_step_policy": "none",
        "current_server_update_policy": SERVER_UPDATE_FEDAVG_MERGED_DELTA,
        "target_server_update_policy": SERVER_UPDATE_FEDMATCH_PARTITIONED,
        "aggregated_partitions": (FEDMATCH_SIGMA_PARTITION, FEDMATCH_PSI_PARTITION),
        "aggregation_weight_policy": "uniform",
    },
)
labels_at_server_policy = FederatedSslServerStepPolicy(
    policy_name=FEDMATCH_LABELS_AT_SERVER_POLICY,
    parameters={
        "scenario": FEDMATCH_SCENARIO_LABELS_AT_SERVER,
        "server_step_policy": "supervised_seed_step",
        "server_update_policy": SERVER_UPDATE_FEDMATCH_PARTITIONED,
        "server_trainable_partition": FEDMATCH_SIGMA_PARTITION,
        "client_aggregated_partitions": (FEDMATCH_PSI_PARTITION,),
        "aggregation_weight_policy": "uniform",
    },
)
helper_context_policy = FederatedSslPeerContextPolicy(
    policy_name=FEDMATCH_HELPER_POLICY,
    parameters={
        "peer_context_policy": FEDMATCH_HELPER_SELECTION_NAME,
        "num_helpers": FEDMATCH_DEFAULT_NUM_HELPERS,
        "refresh_interval": FEDMATCH_DEFAULT_REFRESH_INTERVAL,
        "selection_metric": "model_output_vector_distance",
    },
)


def requires_helper_probability_provider(*, local_ssl_policy_name: str) -> bool:
    """FedMatch agreement objective는 helper weak-view probability를 사용한다."""

    normalized_policy = local_ssl_policy_name.strip().lower().replace("-", "_")
    return normalized_policy == LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT


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
