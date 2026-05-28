"""FL SSL report verification value objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FederatedReportExpectation:
    """이미 생성된 FL SSL report가 맞춰야 하는 실행 metadata 기대값."""

    expected_completed_rounds: int | None = None
    expected_round_budget: int | None = None
    expected_client_count: int | None = None
    expected_seed: int | None = None
    expected_shard_policy_name: str | None = None
    expected_shard_alpha: float | None = None
    expected_split_id: str | None = None
    expected_split_id_contains: str | None = None
    expected_labeled_exposure_policy: str | None = None
    expected_run_control_budget_name: str | None = None
    expected_run_control_output_dir: str | None = None
    expected_fl_method_name: str | None = None
    expected_fl_method_descriptor_name: str | None = None
    expected_fl_method_execution_role: str | None = None
    expected_federated_ssl_method: str | None = None
    expected_ssl_method_implementation_status: str | None = None
    expected_ssl_method_scenario: str | None = None
    expected_ssl_method_local_budget_policy: str | None = None
    expected_ssl_method_parameter_override_status: str | None = None
    expected_ssl_algorithm: str | None = None
    expected_ssl_method: str | None = None
    expected_payload_adapter_kind: str | None = None
    expected_update_family: str | None = None
    expected_aggregation: str | None = None
    expected_server_update_policy: str | None = None
    expected_update_partition_policy: str | None = None
    expected_aggregation_weight_policy: str | None = None
    expected_peer_context_policy: str | None = None
    expected_local_ssl_policy: str | None = None
    expected_delta_format: str | None = None
    expected_round_record_count: int | None = None
    expected_round_update_count: int | None = None
    expected_round_update_count_matches_client_count: bool = False
    expected_shared_update_count: int | None = None
    expected_shared_update_count_matches_round_updates: bool = False
    expect_server_owned_update_artifacts: bool = False
    expect_partitioned_update_artifact_refs: bool = False
    expect_no_agent_local_update_refs: bool = False
    expect_peft_encoder_aggregate_snapshot: bool = False
    expected_communication_estimate_schema_version: str | None = None
    expect_partitioned_sparse_s2c_estimates: bool = False
    expected_embedding_metadata_status: str | None = None
    expected_embedding_backend: str | None = None
    expected_embedding_model_id: str | None = None
    expected_embedding_device: str | None = None
    expected_embedding_local_files_only: bool | None = None
    expected_local_trainer_metadata_status: str | None = None
    expected_local_trainer_device: str | None = None
    expected_local_trainer_local_files_only: bool | None = None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """검증 대상 artifact와 발견된 오류 목록."""

    artifact: str
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors
