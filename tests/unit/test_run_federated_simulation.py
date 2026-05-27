"""run_federated_simulation 스크립트 unit tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from methods.adaptation.query_classifier_adaptation.local_training_budget import (
    build_labeled_anchored_query_ssl_batch_plan,
    build_query_ssl_local_step_plan,
)
from methods.adaptation.text_classifier.peft_encoder import (
    evaluation as peft_encoder_evaluation,
)
from methods.adaptation.text_classifier.peft_encoder.config import (
    LORA_CLASSIFIER_DELTA_FORMAT_AGENT_LOCAL,
    LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
    LORA_CLASSIFIER_DELTA_FORMAT_SERVER_UPLOADED,
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.text_classifier.peft_encoder.evaluation import (
    LORA_CLASSIFIER_EVALUATOR_NAME,
)
from methods.adaptation.text_classifier.peft_encoder.federated_ssl import (
    supervised_seed_step,
)
from methods.adaptation.text_classifier.peft_encoder.update.delta_artifacts import (
    PeftEncoderDeltaMaterializer,
)
from methods.adaptation.text_classifier.peft_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.evaluation.classification_payload import (
    build_classification_evaluation_payload,
)
from methods.evaluation.classification_report import (
    build_classification_evaluation_report,
)
from methods.evaluation.pseudo_label_quality import PseudoLabelQualitySummary
from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.federated_ssl.capability_axes import SERVER_UPDATE_FEDMATCH_PARTITIONED
from methods.federated_ssl.capability_plan import FederatedSslCapabilityPlan
from methods.federated_ssl.execution_plan import build_federated_ssl_execution_plan
from methods.federated_ssl.fedmatch.original_spec import (
    fedmatch_original_parameter_mapping,
)
from methods.federated_ssl.peer_context import (
    FederatedSslPeerClientSnapshot,
    FederatedSslPeerContext,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters import (
    client_training,
    server_step_execution,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.method_runtime import (
    build_federated_ssl_simulation_runtime,
    build_manual_federated_ssl_simulation_runtime,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.sharding import (
    split_rows_for_federation,
    split_rows_into_client_shards,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.round_loop import (
    _build_next_client_partition_sync_state,
    _build_next_query_ssl_algorithm_sync_state,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
    BootstrappedSimulation,
    ClientPartitionSyncSimulationState,
    ClientRoundExecution,
    QuerySslAlgorithmSyncSimulationState,
)
from scripts.experiments.fl_ssl.federated_simulation.io.resume_checkpoint import (
    load_resume_checkpoint,
    resume_checkpoint_path,
    write_resume_checkpoint,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientRoundSummary,
    FederatedClientPoolSplitConfig,
    FederatedClientShard,
    FederatedDatasetSplit,
    FederatedDiagnosticsConfig,
    FederatedLocalTrainerRuntimeConfig,
    FederatedPeftEncoderRuntimeConfig,
    FederatedQuerySslObjectiveConfig,
    FederatedReportConfig,
    FederatedResumeConfig,
    FederatedRoundRuntimeConfig,
    FederatedSslMethodConfig,
    FederatedValidationConfig,
    SimulationEvaluation,
    SimulationRoundSummary,
    SimulationRunRequest,
)
from scripts.experiments.fl_ssl.federated_simulation.runtime_resources import (
    InMemoryRuntimeResourceCache,
    RoundBaseSnapshotCache,
)
from scripts.experiments.fl_ssl.federated_simulation.simulation import (
    _build_validated_ssl_runtime,
    run_simulation_request,
)
from scripts.runtime_adapters.federated_agent import (
    method_owned_client_round,
    query_ssl_client_round,
)
from scripts.runtime_adapters.federated_agent.artifact_store import (
    SimulationClientArtifactStore,
)
from scripts.runtime_adapters.federated_agent.local_training import (
    QuerySslPeftEncoderClientTrainingResult,
)
from scripts.runtime_adapters.federated_server import peft_encoder_server_step
from scripts.runtime_adapters.federated_server.initial_state_factory import (
    build_initial_shared_state,
)
from scripts.runtime_adapters.federated_server.round_request_mapper import (
    build_federated_training_task_config,
    build_round_open_request,
)
from scripts.runtime_adapters.federated_server.runtime import (
    SimulationServerRuntime,
    resolve_simulation_aggregation_backend_name,
)
from shared.src.contracts.adapter_contract_families.classifier_head import (
    ClassifierHeadState,
)
from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    VectorAdapterState,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_lora_classifier_delta_payload,
    make_peft_classifier_delta_payload,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    LoraClassifierState,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    PeftClassifierState,
)
from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.model_contracts import make_embedding_manifest
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
    make_training_update_envelope,
)
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec


def prepare_delta_materialization(*, output_dir, **kwargs):
    return PeftEncoderDeltaMaterializer(
        artifact_store=SimulationClientArtifactStore(output_dir=output_dir)
    ).prepare(**kwargs)


def _row(query_id: str, text: str, label: str) -> dict[str, str]:
    return {
        "query_id": query_id,
        "text": text,
        "raw_label_scheme": "ourafla_4class.v1",
        "raw_label": label.title(),
        "mapped_label_4": label,
        "locale": "eng_Latn",
        "annotation_source": "test",
        "approved_by": "test",
        "created_at": "2026-03-29T00:00:00+00:00",
    }


def test_round_task_mapper_accepts_federated_ssl_method_step_task_type() -> None:
    config = build_federated_training_task_config(
        task_type="federated_ssl_method_local_step",
        local_epochs=1,
        batch_size=4,
        learning_rate=1e-4,
        max_steps=1,
        min_required_examples=1,
        gradient_clip_norm=0.5,
        objective_config={
            "training_backend_name": "lora_classifier_trainer",
        },
        selection_policy={"max_examples": 8},
    )

    assert config.task_type == TrainingTaskType.FEDERATED_SSL_METHOD_LOCAL_STEP


def test_round_task_mapper_migrates_legacy_fedmatch_task_type() -> None:
    config = build_federated_training_task_config(
        task_type="fedmatch_local_step",
        local_epochs=1,
        batch_size=4,
        learning_rate=1e-4,
        max_steps=1,
        min_required_examples=1,
        gradient_clip_norm=0.5,
        objective_config={
            "training_backend_name": "lora_classifier_trainer",
        },
        selection_policy={"max_examples": 8},
    )

    assert config.task_type == TrainingTaskType.FEDERATED_SSL_METHOD_LOCAL_STEP


def _default_shard_policy() -> FederatedShardPolicyConfig:
    return FederatedShardPolicyConfig(
        name="label_dominant",
        dominant_ratio=0.75,
        client_id_prefix="agent",
    )


def _default_training_task_config(
    *,
    confidence_threshold: float,
    margin_threshold: float,
    max_examples: int,
    gradient_clip_norm: float | None,
    training_backend_name: str = "lora_classifier_trainer",
    privacy_guard_name: str = "noop",
    scorer_backend_name: str = "lora_classifier_logits",
    score_policy_name: str = "top1_probability",
    score_top_k: int | None = None,
    task_type: TrainingTaskType | str = TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING,
    objective_extras: dict[str, str | int | float | bool] | None = None,
) -> object:
    return build_federated_training_task_config(
        task_type=task_type,
        local_epochs=1,
        batch_size=16,
        learning_rate=1e-4,
        max_steps=50,
        min_required_examples=1,
        gradient_clip_norm=gradient_clip_norm,
        objective_config=TrainingObjectiveConfig.from_mapping(
            {
                "training_backend_name": training_backend_name,
                "confidence_threshold": confidence_threshold,
                "margin_threshold": margin_threshold,
                "example_generation_backend_name": "lora_classifier_raw_rows",
                "evidence_backend_name": "lora_classifier_logits",
                "scorer_backend_name": scorer_backend_name,
                "score_policy_name": score_policy_name,
                **({} if score_top_k is None else {"score_top_k": score_top_k}),
                "pseudo_label_algorithm_name": "top1_margin_threshold",
                "acceptance_policy_name": "top1_margin_threshold",
                "privacy_guard_name": privacy_guard_name,
                **(objective_extras or {}),
            }
        ),
        selection_policy=TrainingSelectionPolicy.from_mapping(
            {"max_examples": max_examples}
        ),
    )


def _default_validation_config(
    *,
    confidence_threshold: float,
    margin_threshold: float,
    scorer_backend_name: str = LORA_CLASSIFIER_EVALUATOR_NAME,
    score_policy_name: str | None = None,
    score_top_k: int | None = None,
) -> FederatedValidationConfig:
    return FederatedValidationConfig(
        similarity_name="cosine",
        scorer_backend_name=scorer_backend_name,
        score_policy_name=score_policy_name,
        score_top_k=score_top_k,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )


def _default_lora_validation_config(
    *,
    confidence_threshold: float,
    margin_threshold: float,
) -> FederatedValidationConfig:
    return _default_validation_config(
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        scorer_backend_name=LORA_CLASSIFIER_EVALUATOR_NAME,
        score_policy_name=None,
    )


def _patch_lora_classifier_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_evaluator(**kwargs):
        rows = list(kwargs["rows"])
        labels = [str(label) for label in kwargs["adapter_state"].label_schema]
        actual_labels = [str(row["mapped_label_4"]) for row in rows]
        predicted_labels = list(actual_labels)
        probability = 0.9
        report = build_classification_evaluation_report(
            categories=labels,
            actual_labels=actual_labels,
            predicted_labels=predicted_labels,
            true_probs=[probability for _row in rows],
            top_1_values=[probability for _row in rows],
            margins=[0.8 for _row in rows],
            total_loss=0.1 * len(rows),
            total_rows=len(rows),
        )
        return build_classification_evaluation_payload(
            report=report,
            row_count=len(rows),
            accepted_ratio=1.0,
            loss_kind="cross_entropy_from_lora_classifier_logits",
            score_distribution_kind="lora_classifier_logits_softmax",
            selection_confidence_kind="lora_classifier_top1_probability",
            mean_selection_confidence=float(report["mean_top_1_probability"]),
            mean_selection_margin=float(report["mean_margin_top1_top2"]),
        )

    monkeypatch.setattr(
        peft_encoder_evaluation,
        "evaluate_peft_encoder_simulation_validation_payload",
        _fake_evaluator,
    )


def _patch_query_ssl_lora_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_trainer(**kwargs: object) -> QuerySslPeftEncoderClientTrainingResult:
        training_task = kwargs["training_task"]
        active_state = kwargs["active_adapter_state"]
        client_id = str(kwargs["client_id"])
        scale = 1.0 + (sum(ord(char) for char in client_id) % 7) / 10.0
        labels = list(active_state.label_schema)
        if isinstance(active_state, PeftClassifierState):
            update_payload = make_peft_classifier_delta_payload(
                model_id=training_task.model_id,
                base_model_revision=training_task.model_revision,
                training_scope=training_task.training_scope,
                backbone=_lora_runtime_config().backbone_payload(),
                peft_adapter_config=_lora_runtime_config().peft_adapter_config_payload(),
                label_schema=labels,
                example_count=2,
                peft_parameter_deltas={"lora.test": [0.01 * scale]},
                classifier_head_weight_deltas={
                    label: [0.01 * scale, 0.0] for label in labels
                },
                classifier_head_bias_deltas={label: 0.001 * scale for label in labels},
                delta_format=LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
                mean_confidence=0.8,
                delta_l2_norm=0.1 * scale,
            )
            payload_format = PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT
        else:
            update_payload = make_lora_classifier_delta_payload(
                model_id=training_task.model_id,
                base_model_revision=training_task.model_revision,
                training_scope=training_task.training_scope,
                backbone=_lora_runtime_config().backbone_payload(),
                lora_config=_lora_runtime_config().lora_config_payload(),
                label_schema=labels,
                example_count=2,
                lora_parameter_deltas={"lora.test": [0.01 * scale]},
                classifier_head_weight_deltas={
                    label: [0.01 * scale, 0.0] for label in labels
                },
                classifier_head_bias_deltas={label: 0.001 * scale for label in labels},
                delta_format=LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
                mean_confidence=0.8,
                delta_l2_norm=0.1 * scale,
            )
            payload_format = LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT
        update_envelope = make_training_update_envelope(
            update_id=f"update_{client_id}_{training_task.round_id}",
            round_id=training_task.round_id,
            task_id=training_task.task_id,
            model_id=training_task.model_id,
            base_model_revision=training_task.model_revision,
            training_scope=training_task.training_scope,
            payload_ref=f"client-submission::{client_id}",
            payload_format=payload_format,
            example_count=2,
            client_metrics={
                "delta_l2_norm": 0.1 * scale,
                "mean_confidence": 0.8,
                "mean_margin": 0.2,
                "query_ssl_local_steps": 1.0,
            },
        )
        return QuerySslPeftEncoderClientTrainingResult(
            update_envelope=update_envelope,
            update_payload=update_payload,
            candidate_count=2,
            accepted_count=2,
            local_step_plan=build_query_ssl_local_step_plan(
                labeled_loader_steps=1,
                unlabeled_loader_steps=1,
                uses_labeled_batches=True,
                local_epochs=1,
                max_steps=1,
            ),
            client_metrics=update_envelope.client_metrics,
            pseudo_label_quality=PseudoLabelQualitySummary(
                pseudo_label_confidence_mean=0.8,
                pseudo_label_margin_mean=0.2,
                pseudo_label_correct_count=1,
                pseudo_label_evaluated_count=2,
                accepted_label_distribution={labels[0]: 2} if labels else {},
                rejected_label_distribution={},
            ),
        )

    monkeypatch.setattr(
        query_ssl_client_round,
        "run_query_ssl_peft_encoder_local_training",
        _fake_trainer,
    )


def _default_diagnostics_config() -> FederatedDiagnosticsConfig:
    return FederatedDiagnosticsConfig(dump_dir_name="selection_dumps")


def _default_report_config() -> FederatedReportConfig:
    return FederatedReportConfig(
        schema_version="federated_simulation_report.v1",
        track="fl_ssl_main_comparison",
        table_role="main_comparison",
        labeled_ratio=0.1,
        unlabeled_ratio=0.9,
        seed_count=3,
        primary_metrics=["macro_f1", "worst_client_macro_f1"],
        secondary_metrics=[
            "loss",
            "weighted_f1",
            "balanced_accuracy",
            "worst_category_f1_value",
            "expected_calibration_error",
            "max_calibration_error",
            "communication_cost",
            "per_client_macro_f1_variance",
        ],
    )


def _default_client_pool_split_config() -> FederatedClientPoolSplitConfig:
    return FederatedClientPoolSplitConfig(labeled_ratio=0.1, unlabeled_ratio=0.9)


def _legacy_manual_ssl_method_config() -> FederatedSslMethodConfig:
    return FederatedSslMethodConfig(
        schema_version="federated_ssl_method.v1",
        name="legacy_manual_descriptor",
        display_name="Legacy manual descriptor",
        method_role="legacy_descriptor",
        implementation_status="removed",
        client_step={"task_type": "pseudo_label_self_training"},
        server_step={},
        round_state_exchange={"exchange_name": "none"},
    )


def _default_round_runtime_config(
    *,
    adapter_family_name: str = "lora_classifier",
    update_family_name: str = "peft_text_classifier",
    initial_state_builder: str | None = (
        "methods.adaptation.text_classifier.peft_encoder.runtime_family."
        "build_initial_peft_encoder_state"
    ),
    validation_evaluator: str | None = (
        "methods.adaptation.text_classifier.peft_encoder.evaluation."
        "evaluate_peft_encoder_simulation_validation_payload"
    ),
    final_projection_builder: str | None = (
        "scripts.runtime_adapters.federated_server.peft_encoder_final_projection."
        "build_peft_encoder_final_projection_artifacts"
    ),
    aggregation_backend_name: str = "fedavg",
    classifier_head_bootstrap_logit_scale: float = 8.0,
    lora_classifier: FederatedPeftEncoderRuntimeConfig | None = None,
    peft_classifier: FederatedPeftEncoderRuntimeConfig | None = None,
) -> FederatedRoundRuntimeConfig:
    return FederatedRoundRuntimeConfig(
        adapter_family_name=adapter_family_name,
        aggregation_backend_name=aggregation_backend_name,
        update_family_name=update_family_name,
        initial_state_builder=initial_state_builder,
        validation_evaluator=validation_evaluator,
        final_projection_builder=final_projection_builder,
        classifier_head_bootstrap_logit_scale=classifier_head_bootstrap_logit_scale,
        lora_classifier=(
            lora_classifier
            if lora_classifier is not None
            else (
                _lora_runtime_config()
                if adapter_family_name == "lora_classifier"
                else None
            )
        ),
        peft_classifier=(
            peft_classifier
            if peft_classifier is not None
            else (
                _lora_runtime_config()
                if adapter_family_name == "peft_classifier"
                else None
            )
        ),
    )


def _partitioned_server_update_capability_plan() -> FederatedSslCapabilityPlan:
    return FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy={"name": "all_clients"},
        aggregation_weight_policy={"name": "uniform"},
        labeled_exposure_policy={"name": "shared_client_seed"},
        local_supervision_regime={"name": "client_labeled_and_unlabeled"},
        server_step_policy={"name": "none"},
        peer_context_policy={"name": "fixed_probe_output_knn"},
        update_partition_policy={"name": "partitioned"},
        local_ssl_policy={"name": "fixmatch"},
        server_update_policy={"name": SERVER_UPDATE_FEDMATCH_PARTITIONED},
        query_multiview_source={"name": "materialized_rows"},
    )


def _fedmatch_agreement_capability_plan() -> FederatedSslCapabilityPlan:
    return FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy={"name": "all_clients"},
        aggregation_weight_policy={"name": "uniform"},
        labeled_exposure_policy={"name": "shared_client_seed"},
        local_supervision_regime={"name": "client_labeled_and_unlabeled"},
        server_step_policy={"name": "none"},
        peer_context_policy={"name": "fixed_probe_output_knn"},
        update_partition_policy={"name": "partitioned"},
        local_ssl_policy={"name": "fedmatch_agreement"},
        server_update_policy={"name": SERVER_UPDATE_FEDMATCH_PARTITIONED},
        query_multiview_source={"name": "materialized_rows"},
    )


def _lora_runtime_config() -> FederatedPeftEncoderRuntimeConfig:
    return FederatedPeftEncoderRuntimeConfig(
        training_backend_config=LoraClassifierTrainingBackendConfig(
            backbone_model_id="mixedbread-ai/mxbai-embed-large-v1",
            backbone_revision="main",
            tokenizer_model_id="mixedbread-ai/mxbai-embed-large-v1",
            tokenizer_revision="main",
            pooling="mean",
            max_length=256,
            task_prefix="",
            peft_adapter_name="lora",
            rank=8,
            alpha=16,
            dropout=0.1,
            bias="none",
            target_modules="all-linear",
            use_rslora=False,
        ),
    )


def _default_embedding_spec() -> EmbeddingAdapterSpec:
    return EmbeddingAdapterSpec(
        backend="hash_debug",
        model_id="hash_debug",
        revision="sim",
        hash_dim=32,
    )


def _default_train_rows() -> list[dict[str, str]]:
    return [
        _row("a1", "panic panic", "anxiety"),
        _row("a2", "panic panic", "anxiety"),
        _row("n1", "calm calm", "normal"),
        _row("n2", "calm calm", "normal"),
    ]


def _default_validation_rows() -> list[dict[str, str]]:
    return [_row("va", "panic panic", "anxiety")]


def _default_simulation_request(
    tmp_path: Path,
    *,
    output_name: str = "simulation",
    output_dir: Path | None = None,
    train_rows: list[dict[str, str]] | None = None,
    validation_rows: list[dict[str, str]] | None = None,
    client_count: int = 2,
    rounds: int = 0,
    bootstrap_ratio: float = 0.5,
    seed: int = 7,
    model_id: str = "tracemind-embed-sim",
    training_scope: str = "adapter_only",
    round_runtime_config: FederatedRoundRuntimeConfig | None = None,
    training_task_config: object | None = None,
    validation_config: FederatedValidationConfig | None = None,
    ssl_method_config: FederatedSslMethodConfig | None = None,
    client_pool_split_config: FederatedClientPoolSplitConfig | None = None,
    report_config: FederatedReportConfig | None = None,
    shard_policy: FederatedShardPolicyConfig | None = None,
    execution_plan=None,
    server_step_executor: str | None = None,
    query_ssl_objective_config: FederatedQuerySslObjectiveConfig | None = None,
    local_trainer_runtime_config: FederatedLocalTrainerRuntimeConfig | None = None,
    resume_config: FederatedResumeConfig | None = None,
) -> SimulationRunRequest:
    """테스트가 FL simulation 전체 조립 세부사항을 반복하지 않게 한다."""

    return SimulationRunRequest(
        train_rows=train_rows or _default_train_rows(),
        validation_rows=validation_rows or _default_validation_rows(),
        output_dir=output_dir or tmp_path / output_name,
        client_count=client_count,
        rounds=rounds,
        bootstrap_ratio=bootstrap_ratio,
        seed=seed,
        embedding_spec=_default_embedding_spec(),
        model_id=model_id,
        training_scope=training_scope,
        round_runtime_config=round_runtime_config or _default_round_runtime_config(),
        shard_policy=shard_policy or _default_shard_policy(),
        training_task_config=training_task_config
        or _default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
            objective_extras=_lora_objective_extras(),
        ),
        validation_config=validation_config
        or _default_validation_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
        ),
        diagnostics_config=_default_diagnostics_config(),
        resume_config=resume_config or FederatedResumeConfig(),
        ssl_method_config=ssl_method_config,
        client_pool_split_config=(
            client_pool_split_config or _default_client_pool_split_config()
        ),
        report_config=report_config,
        execution_plan=execution_plan,
        server_step_executor=server_step_executor,
        query_ssl_objective_config=query_ssl_objective_config
        or FederatedQuerySslObjectiveConfig(
            method_name="fixmatch_usb_v1",
            algorithm_name="fixmatch",
            parameters={
                "temperature": 0.5,
                "p_cutoff": 0.95,
                "hard_label": True,
                "lambda_u": 1.0,
                "supervised_loss_weight": 1.0,
                "unlabeled_batch_size": 2,
            },
            strong_view_policy="first_aug",
            unlabeled_batch_size=2,
        ),
        local_trainer_runtime_config=(
            local_trainer_runtime_config or FederatedLocalTrainerRuntimeConfig()
        ),
    )


def _lora_objective_extras(
    *,
    delta_format: str = LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
) -> dict[str, str | int | float | bool]:
    return {
        "lora_classifier.backbone_model_id": "mixedbread-ai/mxbai-embed-large-v1",
        "lora_classifier.backbone_revision": "main",
        "lora_classifier.tokenizer_model_id": "mixedbread-ai/mxbai-embed-large-v1",
        "lora_classifier.tokenizer_revision": "main",
        "lora_classifier.pooling": "mean",
        "lora_classifier.max_length": 256,
        "lora_classifier.task_prefix": "",
        "lora_classifier.peft_adapter_name": "lora",
        "lora_classifier.rank": 8,
        "lora_classifier.alpha": 16,
        "lora_classifier.dropout": 0.1,
        "lora_classifier.bias": "none",
        "lora_classifier.target_modules": "all-linear",
        "lora_classifier.use_rslora": False,
        "lora_classifier.delta_format": delta_format,
        "lora_classifier.artifact_ref_prefix": "agent-local://lora_classifier",
        "lora_classifier.text_metadata_keys": (
            "strong_text,training_text,raw_text,text,weak_text"
        ),
        "lora_classifier.label_schema": "anxiety,depression,normal,suicidal",
    }


def _peft_objective_extras(
    *,
    delta_format: str = LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
) -> dict[str, str | int | float | bool]:
    return {
        key.replace("lora_classifier.", "peft_classifier."): value
        for key, value in _lora_objective_extras(delta_format=delta_format).items()
    } | {
        "peft_classifier.artifact_ref_prefix": "agent-local://peft_classifier",
    }


def test_query_ssl_local_step_plan_uses_epochs_batch_steps_and_cap() -> None:
    capped = build_query_ssl_local_step_plan(
        labeled_loader_steps=3,
        unlabeled_loader_steps=5,
        uses_labeled_batches=True,
        local_epochs=2,
        max_steps=7,
    )
    uncapped = build_query_ssl_local_step_plan(
        labeled_loader_steps=3,
        unlabeled_loader_steps=5,
        uses_labeled_batches=True,
        local_epochs=2,
        max_steps=50,
    )
    unlabeled_only = build_query_ssl_local_step_plan(
        labeled_loader_steps=0,
        unlabeled_loader_steps=4,
        uses_labeled_batches=False,
        local_epochs=3,
        max_steps=99,
    )

    assert capped.full_epoch_steps == 5
    assert capped.total_steps == 7
    assert uncapped.total_steps == 10
    assert unlabeled_only.full_epoch_steps == 4
    assert unlabeled_only.total_steps == 12


def test_labeled_anchored_query_ssl_batch_plan_covers_unlabeled_pool() -> None:
    plan = build_labeled_anchored_query_ssl_batch_plan(
        labeled_count=4096,
        unlabeled_count=20305,
        labeled_batch_size=10,
        local_epochs=1,
    )

    assert plan.labeled_batch_size == 10
    assert plan.unlabeled_batch_size == 50
    assert plan.step_plan.full_epoch_steps == 410
    assert plan.step_plan.total_steps == 410
    assert plan.step_plan.max_steps == 410


def test_supervised_seed_step_publishes_server_state_from_bootstrap_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap_row = _row("seed_1", "server labeled panic", "anxiety")
    active_state = build_initial_shared_state(
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=_lora_runtime_config(),
        ),
        model_id="mxbai-lora-classifier",
        model_revision="sim_rev_0000",
        training_scope="adapter_only",
        embedding_dim=2,
        labels=["anxiety", "normal"],
        updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )
    active = ActiveSimulationState(
        manifest=make_embedding_manifest(
            model_id="mxbai-lora-classifier",
            model_revision="sim_rev_0000",
            artifact_ref="shared_adapter_state::sim_rev_0000",
        ),
        adapter_state=active_state,
    )
    request = _default_simulation_request(
        tmp_path,
        output_dir=tmp_path,
        train_rows=[bootstrap_row],
        validation_rows=[bootstrap_row],
        client_count=1,
        rounds=1,
        bootstrap_ratio=1.0,
        model_id="mxbai-lora-classifier",
        ssl_method_config=FederatedSslMethodConfig(
            schema_version="federated_ssl_method.v1",
            name="fedmatch",
            display_name="FedMatch",
            method_role="method_owned",
            implementation_status="active",
            server_step={"name": "supervised_seed_step"},
            effective_parameters={
                "server_pretrain_epochs": 2,
                "server_epochs": 1,
                "server_batch_size": 3,
            },
        ),
        local_trainer_runtime_config=FederatedLocalTrainerRuntimeConfig(
            device="cpu",
            local_files_only=True,
            cache_dir="hf_cache",
        ),
        server_step_executor=(
            "scripts.runtime_adapters.federated_server.peft_encoder_server_step."
            "run_peft_encoder_supervised_seed_step"
        ),
    )
    capability_plan = FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy={"name": "all_clients"},
        aggregation_weight_policy={"name": "uniform"},
        labeled_exposure_policy={"name": "server_only_seed"},
        local_supervision_regime={"name": "client_unlabeled_only"},
        server_step_policy={"name": "supervised_seed_step"},
        peer_context_policy={"name": "none"},
        update_partition_policy={"name": "partitioned"},
        local_ssl_policy={"name": "fixmatch"},
        server_update_policy={"name": SERVER_UPDATE_FEDMATCH_PARTITIONED},
        query_multiview_source={"name": "materialized_rows"},
    )
    calls: dict[str, object] = {}

    class _AdapterFamily:
        def state_to_payload(self, state: object) -> object:
            calls["state_to_payload"] = state
            return state

    class _RoundManager:
        adapter_family = _AdapterFamily()

    class _StateRepository:
        def __init__(self) -> None:
            self.saved: list[object] = []

        def save_shared_adapter_state(self, state: object) -> None:
            self.saved.append(state)

        def ref_for_revision(self, revision: str) -> str:
            return f"shared_adapter_state::{revision}"

    class _ServerRuntime:
        def __init__(self) -> None:
            self.round_manager = _RoundManager()
            self.state_repository = _StateRepository()
            self.activated: list[object] = []

        def activate_manifest(self, manifest: object) -> object:
            self.activated.append(manifest)
            return manifest

        def publish_shared_adapter_projection(
            self,
            *,
            base_manifest: object,
            next_state: object,
            artifacts: object,
            published_at: object,
            notes: str,
        ) -> object:
            del artifacts, published_at, notes
            self.state_repository.save_shared_adapter_state(next_state)
            next_manifest = base_manifest.model_copy(
                update={
                    "model_revision": next_state.model_revision,
                    "artifact_ref": self.state_repository.ref_for_revision(
                        next_state.model_revision
                    ),
                }
            )
            self.activate_manifest(next_manifest)

            class _Publication:
                def __init__(self, *, next_manifest: object, next_state: object):
                    self.next_manifest = next_manifest
                    self.next_state = next_state

            return _Publication(next_manifest=next_manifest, next_state=next_state)

    server_runtime = _ServerRuntime()

    def _fake_build_model(**_: object) -> tuple[object, object]:
        return object(), object()

    def _fake_materialize(**_: object) -> PeftEncoderMaterializedState:
        return PeftEncoderMaterializedState(
            lora_parameters={"lora.test": [0.0]},
            classifier_head_weights={
                "anxiety": [0.0, 0.0],
                "normal": [0.0, 0.0],
            },
            classifier_head_biases={"anxiety": 0.0, "normal": 0.0},
        )

    def _fake_build_dataloader(**kwargs: object) -> list[str]:
        calls["dataloader_rows"] = list(kwargs["rows"])  # type: ignore[index]
        calls["dataloader_batch_size"] = kwargs["batch_size"]
        return ["server-batch"]

    def _fake_train_classifier(**kwargs: object) -> None:
        calls["train_epochs"] = kwargs["epochs"]
        calls["train_loader"] = kwargs["train_loader"]

    monkeypatch.setattr(
        supervised_seed_step,
        "build_peft_encoder_text_classifier_from_config",
        _fake_build_model,
    )
    monkeypatch.setattr(
        peft_encoder_server_step,
        "materialize_base_peft_encoder_state",
        _fake_materialize,
    )
    monkeypatch.setattr(
        supervised_seed_step,
        "load_peft_encoder_base_parameters_into_model",
        lambda **_: None,
    )
    monkeypatch.setattr(
        supervised_seed_step,
        "build_dataloader",
        _fake_build_dataloader,
    )
    monkeypatch.setattr(
        supervised_seed_step,
        "train_classifier",
        _fake_train_classifier,
    )
    monkeypatch.setattr(
        supervised_seed_step,
        "extract_peft_encoder_parameter_deltas",
        lambda **_: (
            {"lora.test": [0.5]},
            {"anxiety": [0.1, 0.0], "normal": [0.0, -0.1]},
            {"anxiety": 0.01, "normal": -0.01},
        ),
    )

    result = server_step_execution.run_server_step_if_supported(
        request=request,
        bootstrapped=BootstrappedSimulation(
            dataset_split=FederatedDatasetSplit(
                bootstrap_rows=[bootstrap_row],
                client_shards=(),
            ),
            validation_client_shards=(),
            server_runtime=server_runtime,  # type: ignore[arg-type]
            initial_model_revision="sim_rev_0000",
            initial_validation=object(),  # type: ignore[arg-type]
            active=active,
        ),
        active=active,
        capability_plan=capability_plan,
        round_index=1,
    )

    assert result.model_revision == "sim_rev_0001_server_seed"
    assert result.active.adapter_state.model_revision == "sim_rev_0001_server_seed"
    assert result.metrics["server_step_labeled_count"] == 1.0
    assert result.metrics["server_step_epochs"] == 2.0
    assert result.metrics["server_step_batch_size"] == 3.0
    assert calls["dataloader_rows"] == [bootstrap_row]
    assert calls["dataloader_batch_size"] == 3
    assert calls["train_epochs"] == 2
    assert calls["train_loader"] == ["server-batch"]
    assert server_runtime.state_repository.saved == [result.active.adapter_state]
    assert len(server_runtime.activated) == 1


def test_query_ssl_lora_round_passes_client_pools_to_real_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labeled_row = _row("l1", "labeled panic", "anxiety")
    unlabeled_row = _row("u1", "weak panic", "anxiety")
    unlabeled_row["aug_0"] = "strong panic de"
    unlabeled_row["aug_1"] = "strong panic fr"
    shard = FederatedClientShard(
        client_id="agent_01",
        rows=[labeled_row, unlabeled_row],
        labeled_rows=[labeled_row],
        unlabeled_rows=[unlabeled_row],
        client_pool_split_enforced=True,
    )
    training_task = TrainingTask(
        task_id="task_round_0001",
        round_id="round_0001",
        model_id="mxbai-lora-classifier",
        model_revision="sim_rev_0000",
        task_type=TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING,
        training_scope="adapter_only",
        local_epochs=3,
        batch_size=2,
        learning_rate=1e-4,
        max_steps=9,
        objective_config=TrainingObjectiveConfig.from_mapping(
            {
                "training_backend_name": "lora_classifier_trainer",
                "confidence_threshold": 0.0,
                "margin_threshold": 0.0,
                "example_generation_backend_name": "lora_classifier_raw_rows",
                "evidence_backend_name": "lora_classifier_logits",
                "scorer_backend_name": "lora_classifier_logits",
                "pseudo_label_algorithm_name": "top1_margin_threshold",
                "privacy_guard_name": "noop",
                **_lora_objective_extras(
                    delta_format=LORA_CLASSIFIER_DELTA_FORMAT_AGENT_LOCAL
                ),
            }
        ),
        selection_policy=TrainingSelectionPolicy.from_mapping({"max_examples": 4}),
        gradient_clip_norm=1.0,
        min_required_examples=1,
    )
    manifest = make_embedding_manifest(
        model_id="mxbai-lora-classifier",
        model_revision="sim_rev_0000",
        artifact_ref="shared_adapter_state::sim_rev_0000",
    )
    active_state = build_initial_shared_state(
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=_lora_runtime_config(),
        ),
        model_id="mxbai-lora-classifier",
        model_revision="sim_rev_0000",
        training_scope="adapter_only",
        embedding_dim=2,
        labels=["anxiety", "normal"],
        updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )
    delta_plan = prepare_delta_materialization(
        output_dir=tmp_path,
        update_id="update_test",
        training_task=training_task,
        client_id="agent_01",
        delta_format=LORA_CLASSIFIER_DELTA_FORMAT_AGENT_LOCAL,
        artifact_ref_prefix="agent-local://lora_classifier",
        lora_parameter_deltas={"lora.test": [0.1]},
        classifier_head_weight_deltas={
            "anxiety": [0.1, 0.0],
            "normal": [0.0, -0.1],
        },
        classifier_head_bias_deltas={"anxiety": 0.01, "normal": -0.01},
    )
    update_payload = make_lora_classifier_delta_payload(
        model_id="mxbai-lora-classifier",
        base_model_revision="sim_rev_0000",
        training_scope="adapter_only",
        backbone=_lora_runtime_config().backbone_payload(),
        lora_config=_lora_runtime_config().lora_config_payload(),
        label_schema=["anxiety", "normal"],
        example_count=2,
        lora_delta_artifact_ref=delta_plan.lora_delta_artifact_ref,
        classifier_head_delta_artifact_ref=(
            delta_plan.classifier_head_delta_artifact_ref
        ),
        delta_format=LORA_CLASSIFIER_DELTA_FORMAT_AGENT_LOCAL,
        mean_confidence=0.5,
        delta_l2_norm=0.2,
    )
    update_envelope = make_training_update_envelope(
        update_id="update_test",
        round_id="round_0001",
        task_id="task_round_0001",
        model_id="mxbai-lora-classifier",
        base_model_revision="sim_rev_0000",
        training_scope="adapter_only",
        payload_ref="client-submission::update_test",
        payload_format=LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
        example_count=2,
        client_metrics={
            "delta_l2_norm": 0.2,
            "mean_confidence": 0.5,
            "mean_margin": 0.0,
            "query_ssl_local_steps": 3.0,
        },
    )
    trainer_calls: list[dict[str, object]] = []
    previous_algorithm_state = {
        "schema_version": "query_ssl_algorithm_state.v1",
        "algorithm_name": "flexmatch",
        "stateful": True,
        "configured": True,
        "round_marker": 1,
    }
    returned_algorithm_state = {
        "schema_version": "query_ssl_algorithm_state.v1",
        "algorithm_name": "flexmatch",
        "stateful": True,
        "configured": True,
        "round_marker": 2,
    }

    def _fake_query_ssl_trainer(
        **kwargs: object,
    ) -> QuerySslPeftEncoderClientTrainingResult:
        trainer_calls.append(dict(kwargs))
        return QuerySslPeftEncoderClientTrainingResult(
            update_envelope=update_envelope,
            update_payload=update_payload,
            candidate_count=1,
            accepted_count=1,
            local_step_plan=build_query_ssl_local_step_plan(
                labeled_loader_steps=1,
                unlabeled_loader_steps=1,
                uses_labeled_batches=True,
                local_epochs=3,
                max_steps=9,
            ),
            client_metrics=update_envelope.client_metrics,
            pseudo_label_quality=PseudoLabelQualitySummary(
                pseudo_label_confidence_mean=0.96,
                pseudo_label_margin_mean=0.41,
                pseudo_label_correct_count=1,
                pseudo_label_evaluated_count=1,
                accepted_label_distribution={"normal": 1},
                rejected_label_distribution={},
            ),
            query_ssl_algorithm_state=returned_algorithm_state,
        )

    monkeypatch.setattr(
        query_ssl_client_round,
        "run_query_ssl_peft_encoder_local_training",
        _fake_query_ssl_trainer,
    )

    class _ServerRuntime:
        def __init__(self) -> None:
            self.accepted: list[tuple[str, object, object]] = []

        def accept_update(
            self,
            round_id: str,
            update_envelope: object,
            update_payload: object,
        ) -> None:
            self.accepted.append((round_id, update_envelope, update_payload))

    server_runtime = _ServerRuntime()
    request = _default_simulation_request(
        tmp_path,
        train_rows=[labeled_row, unlabeled_row],
        validation_rows=[labeled_row],
        output_dir=tmp_path,
        client_count=1,
        rounds=1,
        bootstrap_ratio=0.0,
        seed=42,
        model_id="mxbai-lora-classifier",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=_lora_runtime_config(),
        ),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
            training_backend_name="lora_classifier_trainer",
            privacy_guard_name="noop",
            objective_extras=_lora_objective_extras(
                delta_format=LORA_CLASSIFIER_DELTA_FORMAT_AGENT_LOCAL
            ),
        ),
        validation_config=_default_validation_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
        ),
        query_ssl_objective_config=FederatedQuerySslObjectiveConfig(
            method_name="fixmatch_usb_v1",
            algorithm_name="fixmatch",
            parameters={
                "temperature": 0.5,
                "p_cutoff": 0.95,
                "hard_label": True,
                "lambda_u": 1.0,
                "supervised_loss_weight": 1.0,
                "unlabeled_batch_size": 2,
            },
            strong_view_policy="first_aug",
            unlabeled_batch_size=2,
        ),
        local_trainer_runtime_config=FederatedLocalTrainerRuntimeConfig(
            device="cpu",
            local_files_only=True,
            cache_dir="hf_cache",
            classifier_dropout=0.1,
        ),
    )
    round_base_snapshot_cache = RoundBaseSnapshotCache()
    execution = client_training.run_client_round(
        request=request,
        bootstrapped=BootstrappedSimulation(
            dataset_split=FederatedDatasetSplit(
                bootstrap_rows=[],
                client_shards=(shard,),
            ),
            validation_client_shards=(),
            server_runtime=server_runtime,  # type: ignore[arg-type]
            initial_model_revision="sim_rev_0000",
            initial_validation=object(),  # type: ignore[arg-type]
            active=ActiveSimulationState(
                manifest=manifest,
                adapter_state=active_state,
            ),
            peer_probe_rows=(labeled_row,),
            round_base_snapshot_cache=round_base_snapshot_cache,
        ),
        active=ActiveSimulationState(
            manifest=manifest,
            adapter_state=active_state,
        ),
        ssl_method_runtime=object(),
        round_id="round_0001",
        shard=shard,
        training_task=training_task,
        capability_plan=request.capability_plan,
        previous_query_ssl_algorithm_state=previous_algorithm_state,
    )

    assert execution.update_submitted is True
    assert execution.summary.candidate_count == 1
    assert execution.summary.accepted_count == 1
    assert execution.summary.pseudo_label_correct_count == 1
    assert execution.summary.pseudo_label_evaluated_count == 1
    assert execution.summary.accepted_label_distribution == {"normal": 1}
    assert server_runtime.accepted
    assert trainer_calls
    assert trainer_calls[0]["labeled_rows"] == [labeled_row]
    assert trainer_calls[0]["unlabeled_rows"] == [unlabeled_row]
    assert trainer_calls[0]["training_task"] is training_task
    assert trainer_calls[0]["query_ssl_config"] is request.query_ssl_objective_config
    assert trainer_calls[0]["active_adapter_state"] is active_state
    assert trainer_calls[0]["round_base_snapshot_cache"] is round_base_snapshot_cache
    assert trainer_calls[0]["persist_agent_local_update"] is False
    assert (
        trainer_calls[0]["initial_query_ssl_algorithm_state"]
        is previous_algorithm_state
    )
    assert execution.query_ssl_algorithm_state is returned_algorithm_state
    assert "lora_config" not in trainer_calls[0]
    accepted_payload = server_runtime.accepted[0][2]
    assert accepted_payload.delta_format == (
        LORA_CLASSIFIER_DELTA_FORMAT_SERVER_UPLOADED
    )
    assert accepted_payload.lora_delta_artifact_ref.startswith("aggregation_artifact::")
    assert accepted_payload.classifier_head_delta_artifact_ref.startswith(
        "aggregation_artifact::"
    )
    assert "agent-local://" not in accepted_payload.model_dump_json()


def test_method_owned_lora_round_uses_method_trainer_before_manual_query_ssl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labeled_row = _row("l1", "labeled panic", "anxiety")
    unlabeled_row = _row("u1", "weak panic", "anxiety")
    unlabeled_row["aug_0"] = "strong panic de"
    unlabeled_row["aug_1"] = "strong panic fr"
    shard = FederatedClientShard(
        client_id="agent_01",
        rows=[labeled_row, unlabeled_row],
        labeled_rows=[labeled_row],
        unlabeled_rows=[unlabeled_row],
        client_pool_split_enforced=True,
    )
    training_task = TrainingTask(
        task_id="task_round_0001",
        round_id="round_0001",
        model_id="mxbai-lora-classifier",
        model_revision="sim_rev_0000",
        task_type=TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING,
        training_scope="adapter_only",
        local_epochs=3,
        batch_size=2,
        learning_rate=1e-4,
        max_steps=9,
        objective_config=TrainingObjectiveConfig.from_mapping(
            {
                "training_backend_name": "lora_classifier_trainer",
                "confidence_threshold": 0.0,
                "margin_threshold": 0.0,
                "example_generation_backend_name": "lora_classifier_raw_rows",
                "evidence_backend_name": "lora_classifier_logits",
                "scorer_backend_name": "lora_classifier_logits",
                "pseudo_label_algorithm_name": "top1_margin_threshold",
                "privacy_guard_name": "noop",
                **_lora_objective_extras(
                    delta_format=LORA_CLASSIFIER_DELTA_FORMAT_INLINE
                ),
            }
        ),
        selection_policy=TrainingSelectionPolicy.from_mapping({"max_examples": 4}),
        gradient_clip_norm=1.0,
        min_required_examples=1,
    )
    manifest = make_embedding_manifest(
        model_id="mxbai-lora-classifier",
        model_revision="sim_rev_0000",
        artifact_ref="shared_adapter_state::sim_rev_0000",
    )
    active_state = build_initial_shared_state(
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=_lora_runtime_config(),
        ),
        model_id="mxbai-lora-classifier",
        model_revision="sim_rev_0000",
        training_scope="adapter_only",
        embedding_dim=2,
        labels=["anxiety", "normal"],
        updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )
    update_payload = make_lora_classifier_delta_payload(
        model_id="mxbai-lora-classifier",
        base_model_revision="sim_rev_0000",
        training_scope="adapter_only",
        backbone=_lora_runtime_config().backbone_payload(),
        lora_config=_lora_runtime_config().lora_config_payload(),
        label_schema=["anxiety", "normal"],
        example_count=2,
        lora_parameter_deltas={"lora.test": [0.1]},
        classifier_head_weight_deltas={
            "anxiety": [0.1, 0.0],
            "normal": [0.0, -0.1],
        },
        classifier_head_bias_deltas={"anxiety": 0.01, "normal": -0.01},
        delta_format=LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
        mean_confidence=0.5,
        delta_l2_norm=0.2,
    )
    update_envelope = make_training_update_envelope(
        update_id="update_test",
        round_id="round_0001",
        task_id="task_round_0001",
        model_id="mxbai-lora-classifier",
        base_model_revision="sim_rev_0000",
        training_scope="adapter_only",
        payload_ref="client-submission::update_test",
        payload_format=LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
        example_count=2,
        client_metrics={
            "delta_l2_norm": 0.2,
            "mean_confidence": 0.5,
            "mean_margin": 0.0,
            "query_ssl_local_steps": 3.0,
            "fedmatch_local_runtime": 1.0,
            "fedmatch_helper_count": 1.0,
            "fedmatch_peer_context_helper_count": 1.0,
            "fedmatch_helper_provider_count": 1.0,
            "fedmatch_missing_helper_snapshot_count": 0.0,
            "fedmatch_materialized_helper_model_count": 1.0,
            "fedmatch_peer_context_refreshed": 1.0,
            "fedmatch_c2s_sparse_upload_value_count": 4.0,
            "fedmatch_s2c_sparse_download_value_count": 2.0,
        },
    )
    peer_snapshot = FederatedSslPeerClientSnapshot(
        client_id="agent_02",
        selection_vector=(0.2, 0.8),
        payload_kind="lora_classifier_materialized_state.v1",
        payload=PeftEncoderMaterializedState(
            lora_parameters={"lora.test": [0.2]},
            classifier_head_weights={
                "anxiety": [0.2, 0.0],
                "normal": [0.0, -0.2],
            },
            classifier_head_biases={"anxiety": 0.02, "normal": -0.02},
        ),
    )
    returned_peer_snapshot = FederatedSslPeerClientSnapshot(
        client_id="agent_01",
        selection_vector=(0.7, 0.3),
        payload_kind="lora_classifier_materialized_state.v1",
        payload=PeftEncoderMaterializedState(
            lora_parameters={"lora.test": [0.3]},
            classifier_head_weights={
                "anxiety": [0.3, 0.0],
                "normal": [0.0, -0.3],
            },
            classifier_head_biases={"anxiety": 0.03, "normal": -0.03},
        ),
    )
    previous_client_partition_parameters = {
        "sigma": PeftEncoderMaterializedState(
            lora_parameters={"lora.test": [0.05]},
            classifier_head_weights={
                "anxiety": [0.05, 0.0],
                "normal": [0.0, -0.05],
            },
            classifier_head_biases={"anxiety": 0.005, "normal": -0.005},
        )
    }
    returned_client_partition_parameters = {
        "sigma": PeftEncoderMaterializedState(
            lora_parameters={"lora.test": [0.15]},
            classifier_head_weights={
                "anxiety": [0.15, 0.0],
                "normal": [0.0, -0.15],
            },
            classifier_head_biases={"anxiety": 0.015, "normal": -0.015},
        )
    }
    previous_algorithm_state = {
        "schema_version": "query_ssl_algorithm_state.v1",
        "algorithm_name": "fixmatch",
        "stateful": False,
    }
    returned_algorithm_state = {
        "schema_version": "query_ssl_algorithm_state.v1",
        "algorithm_name": "fixmatch",
        "stateful": False,
        "round_marker": 2,
    }
    method_calls: list[dict[str, object]] = []

    def _fake_method_trainer(
        **kwargs: object,
    ) -> QuerySslPeftEncoderClientTrainingResult:
        method_calls.append(dict(kwargs))
        return QuerySslPeftEncoderClientTrainingResult(
            update_envelope=update_envelope,
            update_payload=update_payload,
            candidate_count=1,
            accepted_count=1,
            local_step_plan=build_query_ssl_local_step_plan(
                labeled_loader_steps=1,
                unlabeled_loader_steps=1,
                uses_labeled_batches=True,
                local_epochs=3,
                max_steps=9,
            ),
            client_metrics=update_envelope.client_metrics,
            pseudo_label_quality=PseudoLabelQualitySummary(
                pseudo_label_confidence_mean=0.91,
                pseudo_label_margin_mean=0.33,
                pseudo_label_correct_count=1,
                pseudo_label_evaluated_count=1,
                accepted_label_distribution={"anxiety": 1},
                rejected_label_distribution={},
            ),
            peer_client_snapshot=returned_peer_snapshot,
            client_partition_parameters=returned_client_partition_parameters,
            query_ssl_algorithm_state=returned_algorithm_state,
        )

    def _unexpected_query_ssl_trainer(**_kwargs: object) -> None:
        raise AssertionError("manual Query SSL trainer must not run for method-owned.")

    monkeypatch.setattr(
        method_owned_client_round.method_trainer,
        "run_method_owned_peft_encoder_local_training",
        _fake_method_trainer,
    )
    monkeypatch.setattr(
        query_ssl_client_round,
        "run_query_ssl_peft_encoder_local_training",
        _unexpected_query_ssl_trainer,
    )

    class _ServerRuntime:
        def __init__(self) -> None:
            self.accepted: list[tuple[str, object, object]] = []

        def accept_update(
            self,
            round_id: str,
            update_envelope: object,
            update_payload: object,
        ) -> None:
            self.accepted.append((round_id, update_envelope, update_payload))

    server_runtime = _ServerRuntime()
    request = _default_simulation_request(
        tmp_path,
        train_rows=[labeled_row, unlabeled_row],
        validation_rows=[labeled_row],
        output_dir=tmp_path,
        client_count=1,
        rounds=1,
        bootstrap_ratio=0.0,
        seed=42,
        model_id="mxbai-lora-classifier",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=_lora_runtime_config(),
        ),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
            training_backend_name="lora_classifier_trainer",
            privacy_guard_name="noop",
            objective_extras=_lora_objective_extras(
                delta_format=LORA_CLASSIFIER_DELTA_FORMAT_INLINE
            ),
        ),
        validation_config=_default_validation_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
        ),
        ssl_method_config=FederatedSslMethodConfig(
            schema_version="federated_ssl_method.v1",
            name="fedmatch",
            display_name="FedMatch",
            method_role="method_owned",
            implementation_status="partitioned_trainable_state_slice_v1",
            scenario="labels-at-client",
            effective_parameters=fedmatch_original_parameter_mapping(),
        ),
        query_ssl_objective_config=FederatedQuerySslObjectiveConfig(
            method_name="fixmatch_usb_v1",
            algorithm_name="fixmatch",
            parameters={"unlabeled_batch_size": 2},
            strong_view_policy="second_aug",
            unlabeled_batch_size=2,
        ),
        local_trainer_runtime_config=FederatedLocalTrainerRuntimeConfig(
            device="cpu",
            local_files_only=True,
            cache_dir="hf_cache",
            classifier_dropout=0.1,
        ),
    )
    request.capability_plan = _fedmatch_agreement_capability_plan()
    runtime_resource_cache = InMemoryRuntimeResourceCache()
    runtime_resource_cache.set_resource("peft_encoder:helper_model:test", object())
    runtime_resource_cache.set_resource("peft_encoder:backbone_base:test", object())
    runtime_resource_cache.set_resource("lora_classifier:helper_model:test", object())
    runtime_resource_cache.set_resource("lora_classifier:backbone_base:test", object())
    runtime_resource_cache.set_resource("lora_classifier:tokenizer:test", object())
    peer_context = FederatedSslPeerContext(
        client_id="agent_01",
        policy_name="fixed_probe_output_knn",
        round_index_zero_based=0,
        helper_client_ids=("agent_02",),
        refreshed=True,
    )
    round_base_snapshot_cache = RoundBaseSnapshotCache()
    execution = client_training.run_client_round(
        request=request,
        bootstrapped=BootstrappedSimulation(
            dataset_split=FederatedDatasetSplit(
                bootstrap_rows=[],
                client_shards=(shard,),
            ),
            validation_client_shards=(),
            server_runtime=server_runtime,  # type: ignore[arg-type]
            initial_model_revision="sim_rev_0000",
            initial_validation=object(),  # type: ignore[arg-type]
            active=ActiveSimulationState(
                manifest=manifest,
                adapter_state=active_state,
            ),
            peer_probe_rows=(labeled_row,),
            runtime_resource_cache=runtime_resource_cache,
            round_base_snapshot_cache=round_base_snapshot_cache,
        ),
        active=ActiveSimulationState(
            manifest=manifest,
            adapter_state=active_state,
        ),
        ssl_method_runtime=object(),
        round_id="round_0001",
        shard=shard,
        training_task=training_task,
        capability_plan=request.capability_plan,
        peer_context=peer_context,
        peer_snapshots={"agent_02": peer_snapshot},
        previous_client_partition_parameters=previous_client_partition_parameters,
        previous_query_ssl_algorithm_state=previous_algorithm_state,
    )

    assert execution.update_submitted is True
    assert execution.summary.accepted_label_distribution == {"anxiety": 1}
    assert server_runtime.accepted
    assert method_calls
    assert method_calls[0]["ssl_method_config"] is request.ssl_method_config
    assert method_calls[0]["labeled_rows"] == [labeled_row]
    assert method_calls[0]["unlabeled_rows"] == [unlabeled_row]
    assert method_calls[0]["peer_context"] is peer_context
    assert method_calls[0]["peer_snapshots"] == {"agent_02": peer_snapshot}
    assert (
        method_calls[0]["previous_client_partition_parameters"]
        is previous_client_partition_parameters
    )
    assert (
        method_calls[0]["initial_query_ssl_algorithm_state"] is previous_algorithm_state
    )
    assert method_calls[0]["peer_probe_rows"] == (labeled_row,)
    assert method_calls[0]["round_base_snapshot_cache"] is round_base_snapshot_cache
    assert method_calls[0]["strong_view_policy"] == "second_aug"
    assert method_calls[0]["unlabeled_batch_size"] == 2
    assert method_calls[0]["persist_agent_local_update"] is False
    assert runtime_resource_cache.get_resource("peft_encoder:helper_model:test") is None
    assert (
        runtime_resource_cache.get_resource("peft_encoder:backbone_base:test") is None
    )
    assert (
        runtime_resource_cache.get_resource("lora_classifier:helper_model:test") is None
    )
    assert (
        runtime_resource_cache.get_resource("lora_classifier:backbone_base:test")
        is None
    )
    assert (
        runtime_resource_cache.get_resource("lora_classifier:tokenizer:test")
        is not None
    )
    assert "helper_model_cache_release_seconds" in execution.summary.timing_breakdown
    assert execution.summary.method_diagnostics["fedmatch_helper_count"] == (
        pytest.approx(1.0)
    )
    assert execution.summary.method_diagnostics[
        "fedmatch_peer_context_helper_count"
    ] == pytest.approx(1.0)
    assert execution.summary.method_diagnostics[
        "fedmatch_helper_provider_count"
    ] == pytest.approx(1.0)
    assert execution.summary.method_diagnostics[
        "fedmatch_missing_helper_snapshot_count"
    ] == pytest.approx(0.0)
    assert execution.summary.method_diagnostics[
        "fedmatch_materialized_helper_model_count"
    ] == pytest.approx(1.0)
    assert execution.summary.method_diagnostics[
        "fedmatch_peer_context_refreshed"
    ] == pytest.approx(1.0)
    assert execution.summary.method_diagnostics[
        "fedmatch_c2s_sparse_upload_value_count"
    ] == pytest.approx(4.0)
    assert execution.summary.method_diagnostics[
        "fedmatch_s2c_sparse_download_value_count"
    ] == pytest.approx(2.0)
    assert execution.peer_client_snapshot is returned_peer_snapshot
    assert execution.client_partition_snapshot is returned_client_partition_parameters
    assert execution.query_ssl_algorithm_state is returned_algorithm_state


def test_build_next_client_partition_sync_state_keeps_previous_snapshots() -> None:
    old_agent_01_partition = {
        "sigma": PeftEncoderMaterializedState(
            lora_parameters={"lora.test": [0.1]},
            classifier_head_weights={"anxiety": [0.1]},
            classifier_head_biases={"anxiety": 0.01},
        )
    }
    old_agent_02_partition = {
        "sigma": PeftEncoderMaterializedState(
            lora_parameters={"lora.test": [0.2]},
            classifier_head_weights={"anxiety": [0.2]},
            classifier_head_biases={"anxiety": 0.02},
        )
    }
    new_agent_01_partition = {
        "sigma": PeftEncoderMaterializedState(
            lora_parameters={"lora.test": [0.3]},
            classifier_head_weights={"anxiety": [0.3]},
            classifier_head_biases={"anxiety": 0.03},
        )
    }

    next_state = _build_next_client_partition_sync_state(
        previous=ClientPartitionSyncSimulationState(
            client_partition_snapshots={
                "agent_01": old_agent_01_partition,
                "agent_02": old_agent_02_partition,
            }
        ),
        client_executions=(
            ClientRoundExecution(
                summary=ClientRoundSummary(
                    client_id="agent_01",
                    candidate_count=1,
                    accepted_count=1,
                    update_generated=True,
                ),
                update_submitted=True,
                client_partition_snapshot=new_agent_01_partition,
            ),
            ClientRoundExecution(
                summary=ClientRoundSummary(
                    client_id="agent_03",
                    candidate_count=1,
                    accepted_count=0,
                    update_generated=False,
                ),
                update_submitted=False,
            ),
        ),
    )

    assert next_state.snapshot_for_client("agent_01") is new_agent_01_partition
    assert next_state.snapshot_for_client("agent_02") is old_agent_02_partition
    assert next_state.snapshot_for_client("agent_03") == {}


def test_build_next_query_ssl_algorithm_sync_state_keeps_client_local_state() -> None:
    old_agent_01_state = {
        "schema_version": "query_ssl_algorithm_state.v1",
        "algorithm_name": "flexmatch",
        "stateful": True,
        "configured": True,
        "round_marker": 1,
    }
    old_agent_02_state = {
        "schema_version": "query_ssl_algorithm_state.v1",
        "algorithm_name": "freematch",
        "stateful": True,
        "configured": True,
        "round_marker": 1,
    }
    new_agent_01_state = {
        "schema_version": "query_ssl_algorithm_state.v1",
        "algorithm_name": "flexmatch",
        "stateful": True,
        "configured": True,
        "round_marker": 2,
    }

    next_state = _build_next_query_ssl_algorithm_sync_state(
        previous=QuerySslAlgorithmSyncSimulationState(
            client_algorithm_states={
                "agent_01": old_agent_01_state,
                "agent_02": old_agent_02_state,
            }
        ),
        client_executions=(
            ClientRoundExecution(
                summary=ClientRoundSummary(
                    client_id="agent_01",
                    candidate_count=1,
                    accepted_count=1,
                    update_generated=True,
                ),
                update_submitted=True,
                query_ssl_algorithm_state=new_agent_01_state,
            ),
            ClientRoundExecution(
                summary=ClientRoundSummary(
                    client_id="agent_03",
                    candidate_count=1,
                    accepted_count=0,
                    update_generated=False,
                ),
                update_submitted=False,
            ),
        ),
    )

    assert next_state.state_for_client("agent_01") is new_agent_01_state
    assert next_state.state_for_client("agent_02") is old_agent_02_state
    assert next_state.state_for_client("agent_03") == {}


def test_resume_checkpoint_preserves_fedmatch_helper_materialization_metrics(
    tmp_path,
) -> None:
    validation = SimulationEvaluation(
        row_count=1,
        top1_accuracy=1.0,
        accepted_ratio=1.0,
    )
    write_resume_checkpoint(
        output_dir=tmp_path,
        initial_model_revision="sim_rev_0000",
        initial_validation=validation,
        rounds=(
            SimulationRoundSummary(
                round_id="round_0001",
                model_revision="sim_rev_0001",
                update_count=1,
                validation=validation,
                clients=(
                    ClientRoundSummary(
                        client_id="agent_01",
                        candidate_count=2,
                        accepted_count=1,
                        update_generated=True,
                        method_diagnostics={
                            "fedmatch_helper_count": 1.0,
                            "fedmatch_peer_context_helper_count": 2.0,
                            "fedmatch_helper_provider_count": 1.0,
                            "fedmatch_missing_helper_snapshot_count": 1.0,
                            "fedmatch_materialized_helper_model_count": 1.0,
                        },
                    ),
                ),
            ),
        ),
    )

    loaded_client = load_resume_checkpoint(tmp_path).rounds[0].clients[0]

    assert loaded_client.method_diagnostics["fedmatch_helper_count"] == pytest.approx(
        1.0
    )
    assert loaded_client.method_diagnostics[
        "fedmatch_peer_context_helper_count"
    ] == pytest.approx(2.0)
    assert loaded_client.method_diagnostics[
        "fedmatch_helper_provider_count"
    ] == pytest.approx(1.0)
    assert loaded_client.method_diagnostics[
        "fedmatch_missing_helper_snapshot_count"
    ] == pytest.approx(1.0)
    assert loaded_client.method_diagnostics[
        "fedmatch_materialized_helper_model_count"
    ] == pytest.approx(1.0)


def test_split_rows_for_federation_keeps_bootstrap_and_client_data_separate() -> None:
    rows = [
        _row("a1", "panic panic", "anxiety"),
        _row("a2", "panic panic", "anxiety"),
        _row("d1", "sad sad", "depression"),
        _row("d2", "sad sad", "depression"),
        _row("n1", "calm calm", "normal"),
        _row("n2", "calm calm", "normal"),
        _row("s1", "die die", "suicidal"),
        _row("s2", "die die", "suicidal"),
    ]

    split = split_rows_for_federation(
        rows,
        bootstrap_ratio=0.5,
        client_count=2,
        seed=42,
        shard_policy=_default_shard_policy(),
    )

    bootstrap_ids = {row["query_id"] for row in split.bootstrap_rows}
    client_ids = {
        row["query_id"] for shard in split.client_shards for row in shard.rows
    }

    assert bootstrap_ids
    assert client_ids
    assert bootstrap_ids.isdisjoint(client_ids)


def test_split_rows_for_federation_enforces_client_pool_split() -> None:
    rows = [_row(f"a{index}", "panic panic", "anxiety") for index in range(12)] + [
        _row(f"n{index}", "calm calm", "normal") for index in range(12)
    ]

    split = split_rows_for_federation(
        rows,
        bootstrap_ratio=0.25,
        client_count=2,
        seed=42,
        shard_policy=FederatedShardPolicyConfig(
            name="label_dominant",
            dominant_ratio=0.5,
            client_id_prefix="agent",
        ),
        client_pool_split_config=FederatedClientPoolSplitConfig(
            labeled_ratio=0.25,
            unlabeled_ratio=0.75,
        ),
    )

    for shard in split.client_shards:
        labeled_ids = {row["query_id"] for row in shard.labeled_rows}
        unlabeled_ids = {row["query_id"] for row in shard.unlabeled_rows}
        all_ids = {row["query_id"] for row in shard.rows}

        assert shard.client_pool_split_enforced is True
        assert labeled_ids
        assert unlabeled_ids
        assert labeled_ids.isdisjoint(unlabeled_ids)
        assert labeled_ids | unlabeled_ids == all_ids


def test_split_rows_for_federation_supports_configurable_dominant_ratio() -> None:
    rows = [
        _row("a1", "panic panic", "anxiety"),
        _row("a2", "panic panic", "anxiety"),
        _row("a3", "panic panic", "anxiety"),
        _row("a4", "panic panic", "anxiety"),
    ]

    split = split_rows_for_federation(
        rows,
        bootstrap_ratio=0.25,
        client_count=2,
        seed=42,
        shard_policy=FederatedShardPolicyConfig(
            name="label_dominant",
            dominant_ratio=0.5,
            client_id_prefix="agent",
        ),
    )

    shard_sizes = [len(shard.rows) for shard in split.client_shards]
    assert shard_sizes == [2, 1]


def test_split_rows_for_federation_supports_dirichlet_label_skew() -> None:
    rows = [_row(f"a{index}", "panic panic", "anxiety") for index in range(20)] + [
        _row(f"n{index}", "calm calm", "normal") for index in range(20)
    ]

    split = split_rows_for_federation(
        rows,
        bootstrap_ratio=0.2,
        client_count=5,
        seed=42,
        shard_policy=FederatedShardPolicyConfig(
            name="dirichlet_label_skew",
            alpha=0.3,
            client_id_prefix="agent",
        ),
    )
    repeated = split_rows_for_federation(
        rows,
        bootstrap_ratio=0.2,
        client_count=5,
        seed=42,
        shard_policy=FederatedShardPolicyConfig(
            name="dirichlet_label_skew",
            alpha=0.3,
            client_id_prefix="agent",
        ),
    )

    assert len(split.client_shards) == 5
    assert [shard.client_id for shard in split.client_shards] == [
        f"agent_{index:02d}" for index in range(1, 6)
    ]
    assert [
        [row["query_id"] for row in shard.rows] for shard in split.client_shards
    ] == [[row["query_id"] for row in shard.rows] for shard in repeated.client_shards]
    input_ids = {row["query_id"] for row in rows}
    output_ids = {row["query_id"] for row in split.bootstrap_rows} | {
        row["query_id"] for shard in split.client_shards for row in shard.rows
    }
    assert output_ids == input_ids


def test_split_rows_into_client_shards_keeps_all_validation_rows() -> None:
    rows = [_row(f"a{index}", "panic panic", "anxiety") for index in range(4)] + [
        _row(f"n{index}", "calm calm", "normal") for index in range(4)
    ]

    shards = split_rows_into_client_shards(
        rows,
        client_count=3,
        seed=42,
        shard_policy=FederatedShardPolicyConfig(
            name="dirichlet_label_skew",
            alpha=0.3,
            client_id_prefix="agent",
        ),
    )

    assert len(shards) == 3
    assert {row["query_id"] for shard in shards for row in shard.rows} == {
        row["query_id"] for row in rows
    }


def test_federated_training_task_config_reuses_round_task_config() -> None:
    training_task_config = _default_training_task_config(
        confidence_threshold=0.6,
        margin_threshold=0.02,
        max_examples=8,
        gradient_clip_norm=0.5,
    )

    request = build_round_open_request(
        round_id="round_0001",
        training_task_config=training_task_config,
    )

    assert training_task_config.__class__.__name__ == "RoundTaskConfig"
    assert request.local_epochs == training_task_config.local_epochs
    assert request.batch_size == training_task_config.batch_size
    assert request.learning_rate == training_task_config.learning_rate
    assert request.max_steps == training_task_config.max_steps
    assert request.objective_config is training_task_config.objective_config
    assert request.selection_policy is training_task_config.selection_policy


def test_federated_training_task_config_accepts_method_task_type() -> None:
    training_task_config = _default_training_task_config(
        confidence_threshold=0.6,
        margin_threshold=0.02,
        max_examples=8,
        gradient_clip_norm=0.5,
        task_type="feedback_supervised",
    )

    assert training_task_config.task_type == TrainingTaskType.FEEDBACK_SUPERVISED


def test_manual_federated_ssl_simulation_runtime_has_no_method_descriptor() -> None:
    runtime = build_manual_federated_ssl_simulation_runtime()

    assert runtime.runtime_name == "manual_baseline"
    assert runtime.descriptor is None
    assert runtime.training_task_type == "pseudo_label_self_training"

    fedmatch_runtime = build_federated_ssl_simulation_runtime("fedmatch")
    assert fedmatch_runtime.runtime_name == "fedmatch"
    assert fedmatch_runtime.descriptor is not None
    assert fedmatch_runtime.descriptor.requires_custom_client_runtime is True
    assert fedmatch_runtime.descriptor.requires_custom_server_runtime is False

    with pytest.raises(NotImplementedError, match="descriptor is not wired yet"):
        build_federated_ssl_simulation_runtime("paper_method_candidate")


def test_simulation_server_runtime_accepts_no_method_descriptor(tmp_path: Path) -> None:
    runtime = SimulationServerRuntime.build(
        output_dir=tmp_path,
        round_runtime_config=_default_round_runtime_config(),
        method_descriptor=None,
    )

    assert runtime.lifecycle_service.method_descriptor is None


def test_simulation_server_runtime_maps_partitioned_server_update_to_backend() -> None:
    backend_name = resolve_simulation_aggregation_backend_name(
        adapter_family_name="lora_classifier",
        aggregation_backend_name="fedavg",
        capability_plan=_partitioned_server_update_capability_plan(),
    )

    assert backend_name == "partitioned_delta_average"


def test_simulation_server_runtime_rejects_partitioned_policy_for_non_lora_family() -> (
    None
):
    with pytest.raises(ValueError, match="not supported by adapter family"):
        resolve_simulation_aggregation_backend_name(
            adapter_family_name="diagonal_scale",
            aggregation_backend_name="fedavg",
            capability_plan=_partitioned_server_update_capability_plan(),
        )


def test_run_simulation_request_rejects_manual_partitioned_update_until_producer(
    tmp_path: Path,
) -> None:
    request = _default_simulation_request(
        tmp_path,
        output_dir=tmp_path,
        rounds=0,
        model_id="mxbai-lora-classifier",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=_lora_runtime_config(),
        ),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
            training_backend_name="lora_classifier_trainer",
            privacy_guard_name="noop",
            objective_extras=_lora_objective_extras(
                delta_format=LORA_CLASSIFIER_DELTA_FORMAT_INLINE
            ),
        ),
        query_ssl_objective_config=FederatedQuerySslObjectiveConfig(
            method_name="fixmatch_usb_v1",
            algorithm_name="fixmatch",
            parameters={"unlabeled_batch_size": 2},
            strong_view_policy="second_aug",
            unlabeled_batch_size=2,
        ),
    )
    request.capability_plan = _partitioned_server_update_capability_plan()

    with pytest.raises(ValueError, match="partitioned_deltas"):
        run_simulation_request(request)


def test_federated_ssl_runtime_rejects_manual_ssl_method_config() -> None:
    execution_plan = build_federated_ssl_execution_plan(
        fl_method={
            "composition_mode": "manual",
            "manual_axes": {
                "client_ssl_objective": "pseudo_label",
                "server_aggregation": "fedavg",
                "update_family": "diagonal_scale",
            },
        },
        security_policy=None,
        method_descriptor=None,
    )

    with pytest.raises(ValueError, match="manual.*ssl_method_config"):
        _build_validated_ssl_runtime(
            _legacy_manual_ssl_method_config(),
            execution_plan=execution_plan,
        )


def test_manual_execution_plan_rejects_round_state_metric_keys() -> None:
    with pytest.raises(
        ValueError,
        match="required_client_metric_keys",
    ):
        build_federated_ssl_execution_plan(
            fl_method={
                "composition_mode": "manual",
                "manual_axes": {
                    "client_ssl_objective": "pseudo_label",
                    "server_aggregation": "fedavg",
                    "update_family": "diagonal_scale",
                },
                "required_client_metric_keys": ["client_entropy"],
            },
            security_policy=None,
            method_descriptor=None,
        )


def test_run_simulation_request_rejects_training_task_type_descriptor_drift(
    tmp_path,
) -> None:
    request = _default_simulation_request(
        tmp_path,
        output_name="task_type_mismatch",
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
            task_type=TrainingTaskType.FEEDBACK_SUPERVISED,
        ),
    )

    with pytest.raises(ValueError, match="training_task_config.task_type"):
        run_simulation_request(request)


def test_run_simulation_request_rejects_manual_plan_runtime_drift(
    tmp_path,
) -> None:
    execution_plan = build_federated_ssl_execution_plan(
        fl_method={
            "composition_mode": "manual",
            "manual_axes": {
                "client_ssl_objective": "fixmatch",
                "server_aggregation": "fedavg",
                "update_family": "peft_text_classifier",
            },
        },
        security_policy=None,
        method_descriptor=None,
    )
    request = _default_simulation_request(
        tmp_path,
        output_name="manual_plan_mismatch",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="diagonal_scale",
            update_family_name="diagonal_scale",
            aggregation_backend_name="fedavg",
        ),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
        ),
        execution_plan=execution_plan,
    )

    with pytest.raises(ValueError, match="manual fl_method.update_family"):
        run_simulation_request(request)


def test_build_initial_shared_state_supports_lora_classifier_family() -> None:
    state = build_initial_shared_state(
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=_lora_runtime_config(),
        ),
        model_id="mxbai-lora-classifier",
        model_revision="sim_rev_0000",
        training_scope="adapter_only",
        embedding_dim=2,
        labels=["anxiety", "normal"],
        updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )

    assert isinstance(state, LoraClassifierState)
    assert state.adapter_kind == "lora_classifier"
    assert state.label_schema == ["anxiety", "normal"]
    assert state.lora_config.rank == 8
    assert state.apply([3.0, 4.0]) == pytest.approx([0.6, 0.8])


def test_build_initial_shared_state_supports_peft_classifier_family() -> None:
    state = build_initial_shared_state(
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="peft_classifier",
            peft_classifier=_lora_runtime_config(),
        ),
        model_id="mxbai-peft-classifier",
        model_revision="sim_rev_0000",
        training_scope="adapter_only",
        embedding_dim=2,
        labels=["anxiety", "normal"],
        updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )

    assert isinstance(state, PeftClassifierState)
    assert state.adapter_kind == "peft_classifier"
    assert state.schema_version == "peft_classifier_state.v2"
    assert state.label_schema == ["anxiety", "normal"]
    assert state.peft_adapter_config.peft_adapter_name == "lora"
    assert state.peft_adapter_config.parameters["rank"] == 8


def test_build_initial_shared_state_uses_configured_diagonal_scale_builder() -> None:
    state = build_initial_shared_state(
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="diagonal_scale",
            update_family_name="diagonal_scale",
            initial_state_builder=(
                "methods.adaptation.diagonal_scale.initial_state."
                "build_initial_diagonal_scale_state"
            ),
        ),
        model_id="mxbai-diagonal",
        model_revision="sim_rev_0000",
        training_scope="adapter_only",
        embedding_dim=3,
        labels=["anxiety", "normal"],
        updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )

    assert isinstance(state, VectorAdapterState)
    assert state.adapter_kind == "diagonal_scale"
    assert state.dimension_scales == [1.0, 1.0, 1.0]


def test_build_initial_shared_state_uses_configured_linear_head_builder() -> None:
    state = build_initial_shared_state(
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="classifier_head",
            update_family_name="linear_head",
            initial_state_builder=(
                "methods.adaptation.classification.feature_head.bootstrap."
                "build_zero_classifier_head_state"
            ),
        ),
        model_id="mxbai-linear-head",
        model_revision="sim_rev_0000",
        training_scope="head_only",
        embedding_dim=2,
        labels=["normal", "anxiety"],
        updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )

    assert isinstance(state, ClassifierHeadState)
    assert state.adapter_kind == "classifier_head"
    assert state.labels == ("anxiety", "normal")
    assert state.label_weights == {"anxiety": [0.0, 0.0], "normal": [0.0, 0.0]}


def test_build_initial_shared_state_rejects_missing_builder() -> None:
    with pytest.raises(ValueError, match="initial_state_builder is required"):
        build_initial_shared_state(
            round_runtime_config=_default_round_runtime_config(
                adapter_family_name="future_family",
                initial_state_builder=None,
            ),
            model_id="future-model",
            model_revision="sim_rev_0000",
            training_scope="adapter_only",
            embedding_dim=2,
            labels=["anxiety", "normal"],
            updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
        )


def test_build_initial_shared_state_rejects_builder_without_state() -> None:
    with pytest.raises(ValueError, match="initial_state_builder returned no"):
        build_initial_shared_state(
            round_runtime_config=_default_round_runtime_config(
                adapter_family_name="future_family"
            ),
            model_id="future-model",
            model_revision="sim_rev_0000",
            training_scope="adapter_only",
            embedding_dim=2,
            labels=["anxiety", "normal"],
            updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
        )


def test_run_simulation_request_rejects_lora_runtime_objective_drift(
    tmp_path,
) -> None:
    drifted_lora_runtime = FederatedPeftEncoderRuntimeConfig(
        training_backend_config=LoraClassifierTrainingBackendConfig(
            backbone_model_id="mixedbread-ai/mxbai-embed-large-v1",
            backbone_revision="main",
            tokenizer_model_id="mixedbread-ai/mxbai-embed-large-v1",
            tokenizer_revision="main",
            pooling="mean",
            max_length=256,
            task_prefix="",
            peft_adapter_name="lora",
            rank=4,
            alpha=16,
            dropout=0.1,
            bias="none",
            target_modules="all-linear",
            use_rslora=False,
        )
    )
    request = _default_simulation_request(
        tmp_path,
        output_name="lora_drift",
        train_rows=[
            _row("a1", "panic panic", "anxiety"),
            _row("n1", "calm calm", "normal"),
        ],
        model_id="mxbai-lora-classifier",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=drifted_lora_runtime,
        ),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
            training_backend_name="lora_classifier_trainer",
            privacy_guard_name="noop",
            objective_extras=_lora_objective_extras(),
        ),
    )

    with pytest.raises(ValueError, match="LoRA-classifier.*training_task.objective"):
        run_simulation_request(request)


def test_run_simulation_request_rejects_missing_lora_runtime_config(
    tmp_path,
) -> None:
    request = _default_simulation_request(
        tmp_path,
        output_name="missing_lora_runtime_config",
        train_rows=[
            _row("a1", "panic panic", "anxiety"),
            _row("n1", "calm calm", "normal"),
        ],
        model_id="mxbai-lora-classifier",
        round_runtime_config=FederatedRoundRuntimeConfig(
            adapter_family_name="lora_classifier",
            aggregation_backend_name="fedavg",
            update_family_name="peft_text_classifier",
            initial_state_builder=(
                "methods.adaptation.text_classifier.peft_encoder.runtime_family."
                "build_initial_peft_encoder_state"
            ),
            validation_evaluator=(
                "methods.adaptation.text_classifier.peft_encoder.evaluation."
                "evaluate_peft_encoder_simulation_validation_payload"
            ),
            classifier_head_bootstrap_logit_scale=8.0,
            lora_classifier=None,
        ),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
            training_backend_name="lora_classifier_trainer",
            privacy_guard_name="noop",
            objective_extras=_lora_objective_extras(),
        ),
    )

    with pytest.raises(
        ValueError,
        match="lora_classifier round runtime requires lora_classifier bootstrap config",
    ):
        run_simulation_request(request)


def test_run_simulation_request_bootstraps_lora_classifier_profile(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_lora_classifier_evaluator(monkeypatch)
    train_rows = [
        _row("a1", "panic panic", "anxiety"),
        _row("a2", "panic panic", "anxiety"),
        _row("n1", "calm calm", "normal"),
        _row("n2", "calm calm", "normal"),
    ]
    validation_rows = [
        _row("va", "panic panic", "anxiety"),
        _row("vn", "calm calm", "normal"),
    ]
    request = _default_simulation_request(
        tmp_path,
        output_name="lora_simulation_request",
        train_rows=train_rows,
        validation_rows=validation_rows,
        model_id="mxbai-lora-classifier",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=_lora_runtime_config(),
        ),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
            training_backend_name="lora_classifier_trainer",
            privacy_guard_name="noop",
            objective_extras=_lora_objective_extras(),
        ),
        validation_config=_default_lora_validation_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
        ),
    )

    result = run_simulation_request(request)

    assert result.initial_model_revision == "sim_rev_0000"
    assert result.rounds == ()
    assert result.final_validation == result.initial_validation


def test_lora_classifier_validation_rejects_prototype_similarity(tmp_path) -> None:
    request = _default_simulation_request(
        tmp_path,
        output_name="lora_prototype_validation_rejected",
        model_id="mxbai-lora-classifier",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=_lora_runtime_config(),
        ),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
            training_backend_name="lora_classifier_trainer",
            privacy_guard_name="noop",
            objective_extras=_lora_objective_extras(),
        ),
        validation_config=_default_validation_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            scorer_backend_name="prototype_similarity",
            score_policy_name="max_cosine",
        ),
    )

    with pytest.raises(ValueError, match="lora_classifier_eval"):
        run_simulation_request(request)


def test_run_simulation_request_completes_lora_classifier_inline_delta_rounds(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_lora_classifier_evaluator(monkeypatch)
    _patch_query_ssl_lora_trainer(monkeypatch)
    train_rows = [
        _row("a1", "panic panic", "anxiety"),
        _row("a2", "panic panic", "anxiety"),
        _row("a3", "panic panic", "anxiety"),
        _row("d1", "sad sad", "depression"),
        _row("d2", "sad sad", "depression"),
        _row("d3", "sad sad", "depression"),
        _row("n1", "calm calm", "normal"),
        _row("n2", "calm calm", "normal"),
        _row("n3", "calm calm", "normal"),
        _row("s1", "die die", "suicidal"),
        _row("s2", "die die", "suicidal"),
        _row("s3", "die die", "suicidal"),
    ]
    validation_rows = [
        _row("va", "panic panic", "anxiety"),
        _row("vd", "sad sad", "depression"),
        _row("vn", "calm calm", "normal"),
        _row("vs", "die die", "suicidal"),
    ]
    output_dir = tmp_path / "lora_inline_round"
    request = _default_simulation_request(
        tmp_path,
        output_name="lora_inline_round",
        train_rows=train_rows,
        validation_rows=validation_rows,
        client_count=4,
        rounds=2,
        bootstrap_ratio=1 / 3,
        model_id="mxbai-lora-classifier",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=_lora_runtime_config(),
        ),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
            training_backend_name="lora_classifier_trainer",
            privacy_guard_name="noop",
            objective_extras=_lora_objective_extras(),
        ),
        validation_config=_default_lora_validation_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
        ),
    )

    result = run_simulation_request(request)

    assert result.rounds
    assert result.rounds[0].update_count > 0
    assert result.rounds[0].model_revision == "sim_rev_0001"
    assert result.rounds[1].update_count > 0
    assert result.rounds[1].model_revision == "sim_rev_0002"
    update_paths = sorted(
        (output_dir / "main_server" / "shared_adapter_updates" / "versions").glob(
            "*.json"
        )
    )
    assert update_paths
    update_payload = json.loads(update_paths[0].read_text(encoding="utf-8"))
    assert update_payload["delta_format"] == LORA_CLASSIFIER_DELTA_FORMAT_INLINE
    assert update_payload["lora_delta_artifact_ref"] is None
    assert update_payload["classifier_head_delta_artifact_ref"] is None
    assert update_payload["lora_parameter_deltas"]
    assert update_payload["classifier_head_weight_deltas"]
    assert "agent-local://" not in json.dumps(update_payload)
    lora_aggregate_path = (
        output_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "lora_classifier"
        / "sim_rev_0001"
        / "lora_adapter.json"
    )
    head_aggregate_path = (
        output_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "lora_classifier"
        / "sim_rev_0001"
        / "classifier_head.json"
    )
    assert lora_aggregate_path.exists()
    assert head_aggregate_path.exists()
    assert json.loads(lora_aggregate_path.read_text(encoding="utf-8"))[
        "lora_parameters"
    ]
    assert json.loads(head_aggregate_path.read_text(encoding="utf-8"))[
        "classifier_head_weights"
    ]
    second_lora_aggregate_path = (
        output_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "lora_classifier"
        / "sim_rev_0002"
        / "lora_adapter.json"
    )
    second_head_aggregate_path = (
        output_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "lora_classifier"
        / "sim_rev_0002"
        / "classifier_head.json"
    )
    assert second_lora_aggregate_path.exists()
    assert second_head_aggregate_path.exists()
    first_lora_artifact = json.loads(lora_aggregate_path.read_text(encoding="utf-8"))
    first_head_artifact = json.loads(head_aggregate_path.read_text(encoding="utf-8"))
    second_lora_artifact = json.loads(
        second_lora_aggregate_path.read_text(encoding="utf-8")
    )
    second_head_artifact = json.loads(
        second_head_aggregate_path.read_text(encoding="utf-8")
    )
    assert second_lora_artifact != first_lora_artifact
    _assert_vector_mapping_accumulates(
        before=first_lora_artifact["lora_parameters"],
        delta=second_lora_artifact["applied_lora_parameter_deltas"],
        after=second_lora_artifact["lora_parameters"],
    )
    _assert_vector_mapping_accumulates(
        before=first_head_artifact["classifier_head_weights"],
        delta=second_head_artifact["applied_classifier_head_weight_deltas"],
        after=second_head_artifact["classifier_head_weights"],
    )
    _assert_scalar_mapping_accumulates(
        before=first_head_artifact["classifier_head_biases"],
        delta=second_head_artifact["applied_classifier_head_bias_deltas"],
        after=second_head_artifact["classifier_head_biases"],
    )


def test_run_simulation_request_completes_peft_classifier_inline_delta_round(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_lora_classifier_evaluator(monkeypatch)
    _patch_query_ssl_lora_trainer(monkeypatch)
    train_rows = [
        _row("a1", "panic panic", "anxiety"),
        _row("a2", "panic panic", "anxiety"),
        _row("n1", "calm calm", "normal"),
        _row("n2", "calm calm", "normal"),
    ]
    validation_rows = [
        _row("va", "panic panic", "anxiety"),
        _row("vn", "calm calm", "normal"),
    ]
    output_dir = tmp_path / "peft_inline_round"
    request = _default_simulation_request(
        tmp_path,
        output_name="peft_inline_round",
        train_rows=train_rows,
        validation_rows=validation_rows,
        output_dir=output_dir,
        client_count=2,
        rounds=1,
        bootstrap_ratio=0.5,
        model_id="mxbai-peft-classifier",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="peft_classifier",
            peft_classifier=_lora_runtime_config(),
        ),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
            training_backend_name="peft_classifier_trainer",
            privacy_guard_name="noop",
            objective_extras=_peft_objective_extras(),
        ),
        validation_config=_default_lora_validation_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
        ),
    )

    result = run_simulation_request(request)

    assert result.rounds
    assert result.rounds[0].update_count > 0
    update_paths = sorted(
        (output_dir / "main_server" / "shared_adapter_updates" / "versions").glob(
            "*.json"
        )
    )
    assert update_paths
    update_payload = json.loads(update_paths[0].read_text(encoding="utf-8"))
    assert update_payload["adapter_kind"] == "peft_classifier"
    assert update_payload["schema_version"] == "peft_classifier_delta.v2"
    assert update_payload["delta_format"] == LORA_CLASSIFIER_DELTA_FORMAT_INLINE
    assert update_payload["peft_parameter_deltas"]
    assert "lora_parameter_deltas" not in update_payload
    peft_aggregate_path = (
        output_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "peft_classifier"
        / "sim_rev_0001"
        / "peft_adapter.json"
    )
    head_aggregate_path = (
        output_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / "peft_classifier"
        / "sim_rev_0001"
        / "classifier_head.json"
    )
    assert peft_aggregate_path.exists()
    assert head_aggregate_path.exists()


def test_run_simulation_request_resumes_from_completed_round_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_lora_classifier_evaluator(monkeypatch)
    _patch_query_ssl_lora_trainer(monkeypatch)
    train_rows = [
        _row("a1", "panic panic", "anxiety"),
        _row("a2", "panic panic", "anxiety"),
        _row("a3", "panic panic", "anxiety"),
        _row("d1", "sad sad", "depression"),
        _row("d2", "sad sad", "depression"),
        _row("d3", "sad sad", "depression"),
        _row("n1", "calm calm", "normal"),
        _row("n2", "calm calm", "normal"),
        _row("n3", "calm calm", "normal"),
        _row("s1", "die die", "suicidal"),
        _row("s2", "die die", "suicidal"),
        _row("s3", "die die", "suicidal"),
    ]
    validation_rows = [
        _row("va", "panic panic", "anxiety"),
        _row("vd", "sad sad", "depression"),
        _row("vn", "calm calm", "normal"),
        _row("vs", "die die", "suicidal"),
    ]
    output_dir = tmp_path / "resume_round"
    base_request = _default_simulation_request(
        tmp_path,
        train_rows=train_rows,
        validation_rows=validation_rows,
        output_dir=output_dir,
        client_count=4,
        rounds=1,
        bootstrap_ratio=1 / 3,
        model_id="mxbai-lora-classifier",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=_lora_runtime_config(),
        ),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
            training_backend_name="lora_classifier_trainer",
            privacy_guard_name="noop",
            objective_extras=_lora_objective_extras(),
        ),
        validation_config=_default_lora_validation_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
        ),
    )

    first_result = run_simulation_request(base_request)

    assert first_result.rounds[-1].model_revision == "sim_rev_0001"
    assert resume_checkpoint_path(output_dir).exists()
    assert load_resume_checkpoint(output_dir).completed_round_count == 1

    resumed_result = run_simulation_request(
        _default_simulation_request(
            tmp_path,
            train_rows=train_rows,
            validation_rows=validation_rows,
            output_dir=output_dir,
            client_count=4,
            rounds=2,
            bootstrap_ratio=1 / 3,
            model_id="mxbai-lora-classifier",
            round_runtime_config=_default_round_runtime_config(
                adapter_family_name="lora_classifier",
                lora_classifier=_lora_runtime_config(),
            ),
            training_task_config=base_request.training_task_config,
            validation_config=base_request.validation_config,
            resume_config=FederatedResumeConfig(
                checkpoint_enabled=True,
                enabled=True,
                run_dir=str(output_dir),
            ),
        )
    )

    resumed_revisions = [
        round_summary.model_revision for round_summary in resumed_result.rounds
    ]
    assert resumed_revisions == ["sim_rev_0001", "sim_rev_0002"]
    assert load_resume_checkpoint(output_dir).completed_round_count == 2


def test_run_simulation_request_rejects_local_round_family_mismatch(
    tmp_path,
) -> None:
    request = _default_simulation_request(
        tmp_path,
        output_name="mismatch_simulation_request",
        train_rows=[
            _row("a1", "panic panic", "anxiety"),
            _row("n1", "calm calm", "normal"),
        ],
        model_id="mxbai-lora-classifier",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=_lora_runtime_config(),
        ),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
            training_backend_name="diagonal_scale_heuristic",
            privacy_guard_name="diagonal_scale_clip_only",
            scorer_backend_name="diagonal_scale_logits",
            objective_extras={},
        ),
    )

    with pytest.raises(ValueError, match="local_update_profile.*round_runtime"):
        run_simulation_request(request)


def test_run_simulation_completes_one_round_with_small_fixture(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_lora_classifier_evaluator(monkeypatch)
    _patch_query_ssl_lora_trainer(monkeypatch)
    train_rows = [
        _row("a1", "panic panic", "anxiety"),
        _row("a2", "panic panic", "anxiety"),
        _row("a3", "panic panic", "anxiety"),
        _row("d1", "sad sad", "depression"),
        _row("d2", "sad sad", "depression"),
        _row("d3", "sad sad", "depression"),
        _row("n1", "calm calm", "normal"),
        _row("n2", "calm calm", "normal"),
        _row("n3", "calm calm", "normal"),
        _row("s1", "die die", "suicidal"),
        _row("s2", "die die", "suicidal"),
        _row("s3", "die die", "suicidal"),
    ]
    validation_rows = [
        _row("va", "panic panic", "anxiety"),
        _row("vd", "sad sad", "depression"),
        _row("vn", "calm calm", "normal"),
        _row("vs", "die die", "suicidal"),
    ]

    result = run_simulation_request(
        _default_simulation_request(
            tmp_path,
            train_rows=train_rows,
            validation_rows=validation_rows,
            output_dir=tmp_path / "simulation",
            client_count=4,
            rounds=1,
            bootstrap_ratio=1 / 3,
            report_config=_default_report_config(),
        )
    )

    assert result.rounds
    assert result.rounds[0].update_count > 0
    assert result.rounds[0].model_revision == "sim_rev_0001"
    assert result.report_path is not None
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert report["track"] == "fl_ssl_main_comparison"
    assert report["table_role"] == "main_comparison"
    assert report["must_not_merge_with"] == ["central_ssl_control"]
    assert report["protocol"]["round_budget"] == 1
    assert report["protocol"]["embedding_adapter"]["backend"] == "hash_debug"
    assert report["protocol"]["embedding_adapter"]["model_id"] == "hash_debug"
    assert report["protocol"]["local_trainer_runtime"]["metadata_status"] == (
        "recorded"
    )
    assert (
        report["protocol"]["artifact_persistence"]["persist_agent_local_updates"]
        is False
    )
    assert report["protocol"]["ssl_method"]["metadata_status"] == "not_applicable"
    assert report["protocol"]["ssl_method"]["reason"] == "manual_composition"
    assert report["protocol"]["fl_method"]["descriptor_name"] is None
    assert report["protocol"]["fl_method"]["execution_role"] == "manual_baseline"
    assert report["protocol"]["labeled_unlabeled_split"]["status"] == (
        "enforced_by_client_pool_split"
    )
    assert report["protocol"]["labeled_unlabeled_split"]["label_distribution"]
    assert report["protocol"]["labeled_unlabeled_split"]["clients"][0][
        "label_distribution"
    ]
    assert report["protocol"]["labeled_unlabeled_split"]["min_client_size"] >= 0
    assert (
        report["protocol"]["labeled_unlabeled_split"]["max_client_size"]
        >= (report["protocol"]["labeled_unlabeled_split"]["min_client_size"])
    )
    assert "label_skew_summary" in report["protocol"]["labeled_unlabeled_split"]
    assert (
        report["protocol"]["labeled_unlabeled_split"]["actual_labeled_count"]
        + report["protocol"]["labeled_unlabeled_split"]["actual_unlabeled_count"]
        > 0
    )
    assert report["protocol"]["local_update_budget"]["local_epochs"] == 1
    assert report["metrics"]["primary"]["macro_f1"] == (
        result.final_validation.macro_f1
    )
    assert report["metrics"]["secondary"]["loss"] == result.final_validation.loss
    assert "weighted_f1" in report["metrics"]["secondary"]
    assert "max_calibration_error" in report["metrics"]["secondary"]
    assert report["metrics"]["final_validation"]["loss_kind"] == (
        "cross_entropy_from_lora_classifier_logits"
    )
    assert report["metrics"]["final_validation"]["score_distribution_kind"] == (
        "lora_classifier_logits_softmax"
    )
    assert report["metrics"]["round_progression"]["round_count"] == 1
    assert (
        report["metrics"]["round_progression"]["early_stop_candidate"]["status"]
        == "insufficient_rounds"
    )
    assert report["rounds"][0]["round_index"] == 1
    assert "accepted_ratio" in report["rounds"][0]["clients"][0]
    assert "delta_l2_norm" in report["rounds"][0]["clients"][0]
    assert "timing_breakdown" in report["rounds"][0]["clients"][0]
    assert (
        report["rounds"][0]["delta_from_previous_round"]["macro_f1_delta"]
        == report["rounds"][0]["delta_from_initial"]["macro_f1_delta"]
    )
    assert report["diagnostics"]["aggregation"]["weight_basis"] == (
        "update_envelope.example_count"
    )
    assert report["diagnostics"]["aggregation"]["rounds"][0]["update_count"] == (
        result.rounds[0].update_count
    )
    assert (
        "aggregation_weight_summary"
        in (report["diagnostics"]["aggregation"]["rounds"][0])
    )
    assert (
        report["diagnostics"]["aggregation"]["rounds"][0]["total_aggregation_examples"]
        >= 0
    )
    assert (
        "zero_update_client_count" in report["diagnostics"]["aggregation"]["rounds"][0]
    )
    assert (
        "delta_l2_norm_summary" in (report["diagnostics"]["aggregation"]["rounds"][0])
    )
    pseudo_label_quality = report["diagnostics"]["pseudo_label_quality"]
    assert (
        pseudo_label_quality["summary"]["candidate_count"]
        >= (pseudo_label_quality["summary"]["accepted_count"])
    )
    assert (
        pseudo_label_quality["summary"]["pseudo_label_accuracy_basis"]
        == "accepted_candidates_with_simulation_labels"
    )
    assert "accepted_label_distribution" in pseudo_label_quality["summary"]
    assert "rejected_label_distribution" in pseudo_label_quality["summary"]
    assert (
        report["metrics"]["secondary"]["communication_cost"]["unit"]
        == "client_update_envelopes"
    )
    assert (
        "timing_breakdown_summary"
        in report["metrics"]["secondary"]["communication_cost"]
    )
    client_validation = report["metrics"]["client_validation"]
    assert client_validation["evaluated_client_count"] > 0
    assert "macro_f1_std" in client_validation
    assert "loss_std" in client_validation
    assert "fairness_gap" in client_validation
    assert client_validation["clients"][0]["client_train_size"] is not None
    assert client_validation["clients"][0]["client_labeled_count"] is not None
    assert client_validation["clients"][0]["client_unlabeled_count"] is not None
    assert "client_accepted_ratio" in client_validation["clients"][0]
    assert "client_update_generated" in client_validation["clients"][0]
    assert "delta_l2_norm_status" in client_validation["clients"][0]
    assert "pseudo_label_accuracy" in client_validation["clients"][0]
    assert "accepted_label_distribution" in client_validation["clients"][0]
    assert "client_timing_breakdown_summary" in client_validation["clients"][0]


def test_run_simulation_request_preserves_typed_boundary(tmp_path, monkeypatch) -> None:
    _patch_lora_classifier_evaluator(monkeypatch)
    train_rows = [
        _row("a1", "panic panic", "anxiety"),
        _row("a2", "panic panic", "anxiety"),
        _row("d1", "sad sad", "depression"),
        _row("d2", "sad sad", "depression"),
        _row("n1", "calm calm", "normal"),
        _row("n2", "calm calm", "normal"),
        _row("s1", "die die", "suicidal"),
        _row("s2", "die die", "suicidal"),
    ]
    validation_rows = [
        _row("va", "panic panic", "anxiety"),
        _row("vd", "sad sad", "depression"),
        _row("vn", "calm calm", "normal"),
        _row("vs", "die die", "suicidal"),
    ]
    request = _default_simulation_request(
        tmp_path,
        output_name="simulation_request",
        train_rows=train_rows,
        validation_rows=validation_rows,
    )

    result = run_simulation_request(request)

    assert result.initial_model_revision == "sim_rev_0000"
    assert result.rounds == ()
    assert result.final_validation == result.initial_validation
    assert len(result.client_evaluations) == 2


def test_run_simulation_accepts_hydra_style_detail_configs(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_lora_classifier_evaluator(monkeypatch)
    _patch_query_ssl_lora_trainer(monkeypatch)
    train_rows = [
        _row("a1", "panic panic", "anxiety"),
        _row("a2", "panic panic", "anxiety"),
        _row("a3", "panic panic", "anxiety"),
        _row("d1", "sad sad", "depression"),
        _row("d2", "sad sad", "depression"),
        _row("d3", "sad sad", "depression"),
        _row("n1", "calm calm", "normal"),
        _row("n2", "calm calm", "normal"),
        _row("n3", "calm calm", "normal"),
        _row("s1", "die die", "suicidal"),
        _row("s2", "die die", "suicidal"),
        _row("s3", "die die", "suicidal"),
    ]
    validation_rows = [
        _row("va", "panic panic", "anxiety"),
        _row("vd", "sad sad", "depression"),
        _row("vn", "calm calm", "normal"),
        _row("vs", "die die", "suicidal"),
    ]

    result = run_simulation_request(
        _default_simulation_request(
            tmp_path,
            train_rows=train_rows,
            validation_rows=validation_rows,
            output_dir=tmp_path / "simulation",
            client_count=4,
            rounds=1,
            bootstrap_ratio=1 / 3,
            shard_policy=FederatedShardPolicyConfig(
                name="label_dominant",
                dominant_ratio=0.5,
                client_id_prefix="agent",
            ),
            training_task_config=_default_training_task_config(
                confidence_threshold=0.0,
                margin_threshold=0.0,
                max_examples=4,
                gradient_clip_norm=1.0,
                score_policy_name="topk_mean_cosine",
                score_top_k=1,
            ),
            validation_config=_default_validation_config(
                confidence_threshold=0.0,
                margin_threshold=0.0,
                score_policy_name="topk_mean_cosine",
                score_top_k=1,
            ),
        )
    )

    assert result.rounds
    assert result.rounds[0].update_count > 0


def _assert_vector_mapping_accumulates(
    *,
    before: Mapping[str, object],
    delta: Mapping[str, object],
    after: Mapping[str, object],
) -> None:
    for key in sorted(set(before) | set(delta)):
        before_values = _sequence_values(before.get(key, []))
        delta_values = _sequence_values(delta.get(key, []))
        after_values = _sequence_values(after[key])
        if not before_values:
            assert after_values == pytest.approx(delta_values)
            continue
        if not delta_values:
            assert after_values == pytest.approx(before_values)
            continue
        assert after_values == pytest.approx(
            [
                before_value + delta_value
                for before_value, delta_value in zip(
                    before_values,
                    delta_values,
                    strict=True,
                )
            ]
        )


def _assert_scalar_mapping_accumulates(
    *,
    before: Mapping[str, object],
    delta: Mapping[str, object],
    after: Mapping[str, object],
) -> None:
    for key in sorted(set(before) | set(delta)):
        assert float(after[key]) == pytest.approx(
            float(before.get(key, 0.0)) + float(delta.get(key, 0.0))
        )


def _sequence_values(value: object) -> list[float]:
    assert isinstance(value, Sequence)
    return [float(item) for item in value]
