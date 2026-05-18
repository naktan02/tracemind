"""FL simulation report payload builder."""

from __future__ import annotations

from collections.abc import Mapping

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.federated_ssl.execution_plan import FederatedSslExecutionPlan
from scripts.experiments.fl_ssl.federated_simulation.io.aggregation_diagnostics import (
    build_aggregation_diagnostics,
)
from scripts.experiments.fl_ssl.federated_simulation.io.client_report import (
    build_client_metric_summary,
)
from scripts.experiments.fl_ssl.federated_simulation.io.protocol_payload import (
    build_protocol_payload,
)
from scripts.experiments.fl_ssl.federated_simulation.io.report_metrics import (
    build_communication_cost_summary,
    build_pseudo_label_quality_diagnostics,
    evaluation_to_payload,
)
from scripts.experiments.fl_ssl.federated_simulation.io.round_report import (
    build_round_payloads,
    build_round_progression,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientPoolSplitConfig,
    FederatedDatasetSplit,
    FederatedDataSourceConfig,
    FederatedLocalTrainerRuntimeConfig,
    FederatedReportConfig,
    FederatedRoundRuntimeConfig,
    FederatedSslMethodConfig,
    FederatedValidationConfig,
    SimulationResult,
)
from scripts.runtime_adapters.federated_server.task_config_surface import (
    FederatedTrainingTaskConfig,
)
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec


class SimulationReportBuilder:
    """FL SSL main comparison report schema payload를 조립한다."""

    def build_payload(
        self,
        *,
        result: SimulationResult,
        report_config: FederatedReportConfig,
        client_count: int,
        round_budget: int,
        bootstrap_ratio: float,
        seed: int,
        run_budget_name: str | None = None,
        run_output_dir: str | None = None,
        shard_policy: FederatedShardPolicyConfig,
        dataset_split: FederatedDatasetSplit,
        ssl_method_config: FederatedSslMethodConfig | None,
        client_pool_split_config: FederatedClientPoolSplitConfig | None,
        training_task_config: FederatedTrainingTaskConfig,
        validation_config: FederatedValidationConfig,
        round_runtime_config: FederatedRoundRuntimeConfig,
        execution_plan: FederatedSslExecutionPlan | None = None,
        data_source_config: FederatedDataSourceConfig | None = None,
        embedding_spec: EmbeddingAdapterSpec | None = None,
        local_trainer_runtime_config: FederatedLocalTrainerRuntimeConfig | None = None,
    ) -> dict[str, object]:
        client_metric_summary = build_client_metric_summary(
            result=result,
            dataset_split=dataset_split,
        )
        communication_cost = build_communication_cost_summary(result)
        round_progression = build_round_progression(result)
        primary_metrics = {
            "macro_f1": result.final_validation.macro_f1,
            "worst_client_macro_f1": client_metric_summary["worst_client_macro_f1"],
        }
        secondary_metrics = {
            "loss": result.final_validation.loss,
            "weighted_f1": result.final_validation.weighted_f1,
            "balanced_accuracy": result.final_validation.balanced_accuracy,
            "worst_category_f1_value": (
                result.final_validation.worst_category_f1_value
            ),
            "max_calibration_error": result.final_validation.max_calibration_error,
            "expected_calibration_error": (
                result.final_validation.expected_calibration_error
            ),
            "communication_cost": communication_cost,
            "per_client_macro_f1_variance": (
                client_metric_summary["macro_f1_variance"]
            ),
        }
        _require_report_metric_names_present(
            configured_metric_names=report_config.primary_metrics,
            payload=primary_metrics,
            section_name="primary",
        )
        _require_report_metric_names_present(
            configured_metric_names=report_config.secondary_metrics,
            payload=secondary_metrics,
            section_name="secondary",
        )
        return {
            "schema_version": report_config.schema_version,
            "track": report_config.track,
            "table_role": report_config.table_role,
            "must_not_merge_with": ["central_ssl_control"],
            "protocol": build_protocol_payload(
                result=result,
                report_config=report_config,
                client_count=client_count,
                round_budget=round_budget,
                bootstrap_ratio=bootstrap_ratio,
                seed=seed,
                run_budget_name=run_budget_name,
                run_output_dir=run_output_dir,
                shard_policy=shard_policy,
                dataset_split=dataset_split,
                ssl_method_config=ssl_method_config,
                client_pool_split_config=client_pool_split_config,
                training_task_config=training_task_config,
                validation_config=validation_config,
                round_runtime_config=round_runtime_config,
                execution_plan=execution_plan,
                data_source_config=data_source_config,
                embedding_spec=embedding_spec,
                local_trainer_runtime_config=local_trainer_runtime_config,
            ),
            "diagnostics": {
                "round_progression": round_progression,
                "aggregation": build_aggregation_diagnostics(result),
                "pseudo_label_quality": build_pseudo_label_quality_diagnostics(result),
                "communication_cost": communication_cost,
            },
            "metrics": {
                "primary": primary_metrics,
                "secondary": secondary_metrics,
                "primary_metric_names": list(report_config.primary_metrics),
                "secondary_metric_names": list(report_config.secondary_metrics),
                "initial_validation": evaluation_to_payload(result.initial_validation),
                "final_validation": evaluation_to_payload(result.final_validation),
                "client_validation": client_metric_summary,
                "round_progression": round_progression,
            },
            "rounds": build_round_payloads(result),
        }


def _require_report_metric_names_present(
    *,
    configured_metric_names: list[str],
    payload: Mapping[str, object],
    section_name: str,
) -> None:
    missing_names = [
        metric_name
        for metric_name in configured_metric_names
        if metric_name not in payload
    ]
    if missing_names:
        raise ValueError(
            f"report.{section_name}_metrics contains metric names missing from "
            f"metrics.{section_name}: {missing_names}"
        )
