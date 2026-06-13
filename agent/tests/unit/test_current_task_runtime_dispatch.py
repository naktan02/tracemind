"""Agent current-task runtime dispatch tests."""

from __future__ import annotations

from agent.src.services.training_runtime.current_task.dispatch import (
    AGENT_RUNTIME_QUERY_SSL_PEFT,
    resolve_current_task_runtime,
)
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfigPayload,
    TrainingSelectionPolicyPayload,
    TrainingTaskPayload,
)


def test_runtime_dispatch_marks_legacy_fssl_context_only_task() -> None:
    plan = resolve_current_task_runtime(
        _query_ssl_task(
            fssl_method="fedmatch",
            fssl_context={"schema_version": "fssl_context.v1"},
        )
    )

    assert plan.runtime_name == AGENT_RUNTIME_QUERY_SSL_PEFT
    assert plan.fssl_method == "fedmatch"
    assert plan.uses_legacy_fssl_context_only is True


def test_runtime_dispatch_does_not_mark_snapshot_task_as_legacy() -> None:
    plan = resolve_current_task_runtime(
        _query_ssl_task(
            fssl_method="fedmatch",
            fssl_execution={
                "composition_mode": "method_owned",
                "execution_role": "method_owned",
                "method_name": "fedmatch",
                "descriptor_name": "fedmatch",
                "runtime_surface": {
                    "payload_adapter_kind": "peft_classifier",
                    "update_family_name": "peft_text_encoder",
                    "aggregation_backend_name": "fedavg",
                },
            },
            fssl_capability_plan={
                "client_participation_policy": {"name": "all_clients"},
                "aggregation_weight_policy": {"name": "uniform"},
                "labeled_exposure_policy": {"name": "shared_client_seed"},
                "local_supervision_regime": {
                    "name": "client_labeled_and_unlabeled"
                },
                "server_step_policy": {"name": "none"},
                "server_update_policy": {"name": "fedmatch_partitioned"},
                "peer_context_policy": {"name": "fixed_probe_output_knn"},
                "update_partition_policy": {"name": "partitioned"},
                "local_ssl_policy": {"name": "fedmatch_agreement"},
                "query_multiview_source": {"name": "materialized_rows"},
            },
        )
    )

    assert plan.fssl_method == "fedmatch"
    assert plan.fssl_capability_plan is not None
    assert plan.uses_legacy_fssl_context_only is False


def _query_ssl_task(
    *,
    fssl_method: str | None = None,
    fssl_execution: dict[str, object] | None = None,
    fssl_capability_plan: dict[str, object] | None = None,
    fssl_context: dict[str, object] | None = None,
) -> TrainingTaskPayload:
    return TrainingTaskPayload(
        schema_version="training_task.v1",
        task_id="task_query_ssl",
        round_id="round_query_ssl",
        model_id="tracemind-embed",
        model_revision="rev_query_ssl",
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        max_steps=4,
        objective_config=TrainingObjectiveConfigPayload(
            algorithm_profile_name="peft_classifier_update_v1",
            training_backend_name="peft_classifier_trainer",
            privacy_guard_name="noop",
            extras={
                "query_ssl.method_name": "fixmatch_usb_v1",
                "query_ssl.algorithm_name": "fixmatch",
                "query_ssl.strong_view_policy": "first_aug",
                "query_ssl.unlabeled_batch_size": 8,
            },
        ),
        selection_policy=TrainingSelectionPolicyPayload(),
        fssl_method=fssl_method,
        fssl_execution=fssl_execution,
        fssl_capability_plan=fssl_capability_plan,
        fssl_context=fssl_context,
    )
