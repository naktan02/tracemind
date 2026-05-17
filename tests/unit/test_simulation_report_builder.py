"""FL simulation report builder 계산값 검증."""

from __future__ import annotations

import pytest

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from scripts.experiments.fl_ssl.federated_simulation.io import (
    simulation_report_builder,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientEvaluationSummary,
    ClientRoundSummary,
    FederatedClientPoolSplitConfig,
    FederatedClientShard,
    FederatedDatasetSplit,
    FederatedDataSourceConfig,
    FederatedLocalTrainerRuntimeConfig,
    FederatedReportConfig,
    FederatedRoundRuntimeConfig,
    FederatedSslMethodConfig,
    FederatedValidationConfig,
    SimulationEvaluation,
    SimulationResult,
    SimulationRoundSummary,
)
from scripts.runtime_adapters.federated_server.round_request_mapper import (
    build_federated_training_task_config,
)
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec


def _row(query_id: str, label: str) -> dict[str, str]:
    return {
        "query_id": query_id,
        "text": f"{label} text",
        "mapped_label_4": label,
    }


def _evaluation(
    *,
    macro_f1: float,
    loss: float,
    accepted_ratio: float = 0.5,
    row_count: int = 10,
) -> SimulationEvaluation:
    return SimulationEvaluation(
        row_count=row_count,
        top1_accuracy=macro_f1,
        accepted_ratio=accepted_ratio,
        loss=loss,
        loss_kind="negative_log_likelihood_from_score_distribution",
        accuracy_top_1=macro_f1,
        correct_top_1=round(macro_f1 * row_count),
        macro_f1=macro_f1,
        macro_precision=macro_f1,
        macro_recall=macro_f1,
        weighted_f1=macro_f1,
        balanced_accuracy=macro_f1,
        expected_calibration_error=0.1,
        max_calibration_error=0.2,
        score_distribution_kind="softmax_raw_scores_temperature_1.0",
    )


def _report_config() -> FederatedReportConfig:
    return FederatedReportConfig(
        schema_version="federated_simulation_report.v1",
        track="fl_ssl_main_comparison",
        table_role="main_comparison",
        labeled_ratio=0.1,
        unlabeled_ratio=0.9,
        seed_count=3,
        primary_metrics=["macro_f1", "worst_client_macro_f1"],
        secondary_metrics=["loss", "communication_cost"],
    )


def _ssl_method_config() -> FederatedSslMethodConfig:
    return FederatedSslMethodConfig(
        schema_version="federated_ssl_method.v1",
        name="fedavg_pseudo_label",
        display_name="FedAvg pseudo-label baseline",
        method_role="baseline",
        implementation_status="active_runtime",
        client_step={"task_type": "pseudo_label_self_training"},
        server_step={"aggregation_backend_name": "fedavg"},
    )


def _training_task_config() -> object:
    return build_federated_training_task_config(
        local_epochs=1,
        batch_size=16,
        learning_rate=1e-4,
        max_steps=50,
        min_required_examples=1,
        gradient_clip_norm=1.0,
        objective_config=TrainingObjectiveConfig.from_mapping(
            {
                "training_backend_name": "diagonal_scale_heuristic",
                "confidence_threshold": 0.0,
                "margin_threshold": 0.0,
            }
        ),
        selection_policy=TrainingSelectionPolicy.from_mapping({"max_examples": 8}),
    )


def _dataset_split() -> FederatedDatasetSplit:
    agent_001_rows = [
        _row("a1", "anxiety"),
        _row("a2", "anxiety"),
        _row("n1", "normal"),
    ]
    agent_002_rows = [
        _row("d1", "depression"),
        _row("d2", "depression"),
    ]
    return FederatedDatasetSplit(
        bootstrap_rows=[_row("b1", "normal")],
        client_shards=(
            FederatedClientShard(
                client_id="agent_001",
                rows=agent_001_rows,
                labeled_rows=agent_001_rows[:1],
                unlabeled_rows=agent_001_rows[1:],
                client_pool_split_enforced=True,
            ),
            FederatedClientShard(
                client_id="agent_002",
                rows=agent_002_rows,
                labeled_rows=agent_002_rows[:1],
                unlabeled_rows=agent_002_rows[1:],
                client_pool_split_enforced=True,
            ),
        ),
    )


def test_simulation_report_builder_computes_round_client_and_split_metrics() -> None:
    result = SimulationResult(
        initial_model_revision="sim_rev_0000",
        initial_prototype_version="proto_sim_0000",
        initial_validation=_evaluation(macro_f1=0.2, loss=0.9),
        final_validation=_evaluation(macro_f1=0.6, loss=0.5),
        rounds=(
            SimulationRoundSummary(
                round_id="round_0001",
                model_revision="sim_rev_0001",
                prototype_version="proto_sim_0001",
                update_count=1,
                validation=_evaluation(macro_f1=0.4, loss=0.8),
                round_time_seconds=1.5,
                total_payload_bytes=100,
                clients=(
                    ClientRoundSummary(
                        client_id="agent_001",
                        candidate_count=10,
                        accepted_count=5,
                        update_generated=True,
                        delta_l2_norm=2.0,
                        aggregation_example_count=5,
                        client_train_time_seconds=0.11,
                        client_payload_bytes=100,
                        pseudo_label_confidence_mean=0.8,
                        pseudo_label_margin_mean=0.2,
                        pseudo_label_correct_count=4,
                        pseudo_label_evaluated_count=5,
                        accepted_label_distribution={"anxiety": 5},
                        rejected_label_distribution={"normal": 5},
                    ),
                    ClientRoundSummary(
                        client_id="agent_002",
                        candidate_count=10,
                        accepted_count=0,
                        update_generated=False,
                        client_train_time_seconds=0.07,
                        pseudo_label_confidence_mean=0.4,
                        pseudo_label_margin_mean=0.1,
                        rejected_label_distribution={"depression": 10},
                    ),
                ),
            ),
            SimulationRoundSummary(
                round_id="round_0002",
                model_revision="sim_rev_0002",
                prototype_version="proto_sim_0002",
                update_count=2,
                validation=_evaluation(macro_f1=0.6, loss=0.5),
                round_time_seconds=2.5,
                total_payload_bytes=200,
                clients=(
                    ClientRoundSummary(
                        client_id="agent_001",
                        candidate_count=10,
                        accepted_count=4,
                        update_generated=True,
                        delta_l2_norm=4.0,
                        aggregation_example_count=4,
                        client_train_time_seconds=0.2,
                        client_payload_bytes=80,
                        pseudo_label_confidence_mean=0.9,
                        pseudo_label_margin_mean=0.3,
                        pseudo_label_correct_count=4,
                        pseudo_label_evaluated_count=4,
                        accepted_label_distribution={"anxiety": 4},
                        rejected_label_distribution={"normal": 6},
                    ),
                    ClientRoundSummary(
                        client_id="agent_002",
                        candidate_count=10,
                        accepted_count=6,
                        update_generated=True,
                        delta_l2_norm=6.0,
                        aggregation_example_count=6,
                        client_train_time_seconds=0.3,
                        client_payload_bytes=120,
                        pseudo_label_confidence_mean=0.7,
                        pseudo_label_margin_mean=0.2,
                        pseudo_label_correct_count=3,
                        pseudo_label_evaluated_count=6,
                        accepted_label_distribution={"depression": 6},
                        rejected_label_distribution={"normal": 4},
                    ),
                ),
            ),
        ),
        client_evaluations=(
            ClientEvaluationSummary(
                client_id="agent_001",
                validation=_evaluation(macro_f1=0.7, loss=0.3),
            ),
            ClientEvaluationSummary(
                client_id="agent_002",
                validation=_evaluation(macro_f1=0.5, loss=0.7),
            ),
        ),
    )

    payload = simulation_report_builder.SimulationReportBuilder().build_payload(
        result=result,
        report_config=_report_config(),
        client_count=2,
        round_budget=2,
        bootstrap_ratio=0.2,
        seed=7,
        shard_policy=FederatedShardPolicyConfig(
            name="label_dominant",
            client_id_prefix="agent",
            dominant_ratio=0.75,
        ),
        dataset_split=_dataset_split(),
        ssl_method_config=_ssl_method_config(),
        client_pool_split_config=FederatedClientPoolSplitConfig(
            labeled_ratio=0.1,
            unlabeled_ratio=0.9,
        ),
        training_task_config=_training_task_config(),
        validation_config=FederatedValidationConfig(
            similarity_name="cosine",
            scorer_backend_name="prototype_similarity",
            score_policy_name="max_cosine",
            confidence_threshold=0.0,
            margin_threshold=0.0,
        ),
        round_runtime_config=FederatedRoundRuntimeConfig(
            adapter_family_name="diagonal_scale",
            aggregation_backend_name="fedavg",
        ),
        embedding_spec=EmbeddingAdapterSpec(
            backend="mxbai",
            model_id="mixedbread-ai/mxbai-embed-large-v1",
            revision="main",
            device="cuda",
            batch_size=16,
            cache_dir="data/cache/hf",
            local_files_only=True,
        ),
        local_trainer_runtime_config=FederatedLocalTrainerRuntimeConfig(
            device="cuda",
            local_files_only=True,
            cache_dir="data/cache/hf",
            trust_remote_code=False,
            classifier_dropout=0.1,
        ),
        data_source_config=FederatedDataSourceConfig(
            source_mode="materialized_client_split",
            split_manifest_path="data/datasets/fl_client_splits/main/manifest.json",
            split_manifest_sha256="abc123",
            split_id="main",
            source_selection={"labeled": "ourafla_reddit"},
            source_jsonl={"labeled": "labeled.jsonl"},
            labeled_policy={"mode": "all"},
            view_schema={
                "weak_text_field": "text",
                "strong_text_fields": ["aug_0", "aug_1"],
            },
            test_jsonl="test.jsonl",
        ),
    )

    second_round_aggregation = payload["diagnostics"]["aggregation"]["rounds"][1]
    assert payload["rounds"][0]["round_index"] == 1
    assert payload["rounds"][0]["global_validation"]["macro_f1"] == pytest.approx(0.4)
    assert payload["rounds"][0]["round_time_seconds"] == pytest.approx(1.5)
    assert payload["rounds"][0]["total_payload_bytes"] == 100
    assert payload["rounds"][1]["delta_from_previous_round"]["loss_reduction"] == (
        pytest.approx(0.3)
    )
    round_progression = payload["diagnostics"]["round_progression"]
    assert len(round_progression["validation_curve"]) == 3
    assert round_progression["best_round"]["round_id"] == "round_0002"
    assert round_progression["best_round"]["selection_metric"] == "macro_f1"
    assert round_progression["validation_curve"][0]["round_index"] == 0
    assert second_round_aggregation["zero_update_client_count"] == 0
    assert second_round_aggregation["total_aggregation_examples"] == 10
    assert second_round_aggregation["aggregation_example_basis"] == (
        "update_envelope.example_count"
    )
    assert second_round_aggregation["aggregation_weight_summary"]["max"] == (
        pytest.approx(0.6)
    )
    assert second_round_aggregation["mean_delta_l2_norm"] == pytest.approx(5.0)
    assert second_round_aggregation["max_delta_l2_norm"] == pytest.approx(6.0)
    assert second_round_aggregation["update_norm_variance"] == pytest.approx(1.0)

    pseudo_label_quality = payload["diagnostics"]["pseudo_label_quality"]
    assert pseudo_label_quality["summary"]["candidate_count"] == 40
    assert pseudo_label_quality["summary"]["accepted_count"] == 15
    assert pseudo_label_quality["summary"]["pseudo_label_accuracy"] == pytest.approx(
        11 / 15
    )
    assert pseudo_label_quality["summary"]["candidate_confidence_mean"] == (
        pytest.approx(0.7)
    )
    assert pseudo_label_quality["summary"]["accepted_label_distribution"] == {
        "anxiety": 9,
        "depression": 6,
    }

    communication_cost = payload["diagnostics"]["communication_cost"]
    assert communication_cost["total_payload_bytes"] == 300
    assert communication_cost["payload_byte_accounting_status"] == "measured"
    assert communication_cost["round_time_seconds"]["mean"] == pytest.approx(2.0)
    assert communication_cost["client_train_time_seconds"]["max"] == pytest.approx(0.3)

    client_validation = payload["metrics"]["client_validation"]
    agent_001_summary = client_validation["clients"][0]
    assert client_validation["fairness_gap"] == pytest.approx(0.2)
    assert client_validation["macro_f1_std"] == pytest.approx(0.1)
    assert client_validation["loss_std"] == pytest.approx(0.2)
    assert agent_001_summary["client_train_size"] == 3
    assert agent_001_summary["client_labeled_count"] == 1
    assert agent_001_summary["client_unlabeled_count"] == 2
    assert agent_001_summary["client_candidate_count"] == 20
    assert agent_001_summary["client_accepted_count"] == 9
    assert agent_001_summary["client_accepted_ratio"] == pytest.approx(0.45)
    assert agent_001_summary["client_payload_bytes"] == 180
    assert agent_001_summary["update_generated_round_count"] == 2
    assert agent_001_summary["client_delta_l2_norm"] == pytest.approx(4.0)
    assert agent_001_summary["mean_delta_l2_norm"] == pytest.approx(3.0)
    assert agent_001_summary["client_train_time_seconds"] == pytest.approx(0.2)
    assert agent_001_summary["mean_client_train_time_seconds"] == pytest.approx(0.155)
    assert agent_001_summary["client_validation_loss"] == pytest.approx(0.3)
    assert agent_001_summary["client_validation_macro_f1"] == pytest.approx(0.7)
    assert agent_001_summary["client_validation_ece"] == pytest.approx(0.1)
    assert agent_001_summary["pseudo_label_accuracy"] == pytest.approx(8 / 9)
    assert agent_001_summary["accepted_label_distribution"] == {"anxiety": 9}

    split = payload["protocol"]["labeled_unlabeled_split"]
    assert split["status"] == "materialized_client_split"
    assert split["labeled_ratio"] == pytest.approx(0.4)
    assert split["unlabeled_ratio"] == pytest.approx(0.6)
    assert split["configured_labeled_ratio"] == pytest.approx(0.1)
    assert split["min_client_size"] == 2
    assert split["max_client_size"] == 3
    assert split["label_skew_summary"]["dominant_label_ratio"]["max"] == 1.0
    fl_data_source = payload["protocol"]["fl_data_source"]
    assert fl_data_source["source_mode"] == "materialized_client_split"
    assert fl_data_source["split_manifest_sha256"] == "abc123"
    assert fl_data_source["labeled_policy"] == {"mode": "all"}
    assert fl_data_source["view_schema"]["strong_text_fields"] == [
        "aug_0",
        "aug_1",
    ]
    embedding_adapter = payload["protocol"]["embedding_adapter"]
    assert embedding_adapter["metadata_status"] == "recorded"
    assert embedding_adapter["backend"] == "mxbai"
    assert embedding_adapter["model_id"] == "mixedbread-ai/mxbai-embed-large-v1"
    assert embedding_adapter["device"] == "cuda"
    local_trainer_runtime = payload["protocol"]["local_trainer_runtime"]
    assert local_trainer_runtime["metadata_status"] == "recorded"
    assert local_trainer_runtime["device"] == "cuda"
    assert local_trainer_runtime["local_files_only"] is True


def test_simulation_report_builder_rejects_unknown_metric_names() -> None:
    report_config = _report_config()
    report_config.primary_metrics.append("missing_metric")
    result = SimulationResult(
        initial_model_revision="sim_rev_0000",
        initial_prototype_version="proto_sim_0000",
        initial_validation=_evaluation(macro_f1=0.2, loss=0.9),
        final_validation=_evaluation(macro_f1=0.6, loss=0.5),
        rounds=(),
        client_evaluations=(),
    )

    with pytest.raises(ValueError, match="report.primary_metrics"):
        simulation_report_builder.SimulationReportBuilder().build_payload(
            result=result,
            report_config=report_config,
            client_count=2,
            round_budget=2,
            bootstrap_ratio=0.2,
            seed=7,
            shard_policy=FederatedShardPolicyConfig(
                name="label_dominant",
                client_id_prefix="agent",
                dominant_ratio=0.75,
            ),
            dataset_split=_dataset_split(),
            ssl_method_config=_ssl_method_config(),
            client_pool_split_config=FederatedClientPoolSplitConfig(
                labeled_ratio=0.1,
                unlabeled_ratio=0.9,
            ),
            training_task_config=_training_task_config(),
            validation_config=FederatedValidationConfig(
                similarity_name="cosine",
                scorer_backend_name="prototype_similarity",
                score_policy_name="max_cosine",
                confidence_threshold=0.0,
                margin_threshold=0.0,
            ),
            round_runtime_config=FederatedRoundRuntimeConfig(
                adapter_family_name="diagonal_scale",
                aggregation_backend_name="fedavg",
            ),
        )
