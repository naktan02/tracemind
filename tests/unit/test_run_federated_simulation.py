"""run_federated_simulation 스크립트 unit tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from methods.adaptation.lora_classifier.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor
from methods.prototype.building.single import (
    SinglePrototypeBuildStrategy,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.evaluation import (
    build_training_examples,
    evaluate_rows,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.method_runtime import (
    build_federated_ssl_simulation_runtime,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.sharding import (
    split_rows_for_federation,
    split_rows_into_client_shards,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientPoolSplitConfig,
    FederatedClientShard,
    FederatedDiagnosticsConfig,
    FederatedLoraClassifierRuntimeConfig,
    FederatedPrototypeRebuildConfig,
    FederatedReportConfig,
    FederatedRoundRuntimeConfig,
    FederatedSslMethodConfig,
    FederatedValidationConfig,
    SimulationRunRequest,
)
from scripts.experiments.fl_ssl.federated_simulation.simulation import (
    _build_validated_ssl_runtime,
    run_simulation,
    run_simulation_request,
)
from scripts.runtime_adapters.federated_agent.scoring_runtime import (
    build_federated_scoring_service,
)
from scripts.runtime_adapters.federated_server.initial_state_factory import (
    build_initial_shared_state,
)
from scripts.runtime_adapters.federated_server.round_request_mapper import (
    build_federated_training_task_config,
    build_round_open_request,
)
from scripts.runtime_adapters.federated_server.runtime import SimulationServerRuntime
from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    VectorAdapterState,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierState,
)
from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec


class _StaticEmbeddingAdapter:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vectors[text]) for text in texts]


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


def _pack_payload() -> PrototypePackPayload:
    return PrototypePackPayload.model_validate(
        {
            "schema_version": "prototype_pack.v1",
            "prototype_version": "proto_test_v1",
            "embedding_model_id": "hash_debug",
            "embedding_model_revision": "main",
            "mapping_version": "ourafla_to_4cat.v1",
            "build_method": "mean_centroid_l2_normalized",
            "distance_metric": "cosine",
            "built_at": "2026-04-02T00:00:00+00:00",
            "categories": {
                "anxiety": [
                    {
                        "prototype_id": "anxiety:single",
                        "centroid": [1.0, 0.0],
                        "sample_count": 2,
                    }
                ],
                "normal": [
                    {
                        "prototype_id": "normal:single",
                        "centroid": [0.0, 1.0],
                        "sample_count": 2,
                    }
                ],
            },
        }
    )


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
    training_backend_name: str = "diagonal_scale_heuristic",
    privacy_guard_name: str = "diagonal_scale_clip_only",
    scorer_backend_name: str = "prototype_similarity",
    score_policy_name: str = "max_cosine",
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
                "example_generation_backend_name": "prototype_rescore",
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
    scorer_backend_name: str = "prototype_similarity",
    score_policy_name: str | None = "max_cosine",
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


def _default_prototype_rebuild_config() -> FederatedPrototypeRebuildConfig:
    return FederatedPrototypeRebuildConfig(
        embedding_backend="simulation",
        mapping_version="ourafla_to_4cat.v1",
        translation_model_id=None,
        translation_model_revision=None,
        translation_direction=None,
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


def _default_ssl_method_config() -> FederatedSslMethodConfig:
    return FederatedSslMethodConfig(
        schema_version="federated_ssl_method.v1",
        name="fedavg_pseudo_label",
        display_name="FedAvg pseudo-label baseline",
        method_role="baseline",
        implementation_status="active_runtime",
        client_step={
            "owner": "agent",
            "task_type": "pseudo_label_self_training",
            "custom_method_runtime_required": False,
        },
        server_step={
            "owner": "main_server",
            "aggregation_backend_name": "fedavg",
            "custom_round_policy_required": False,
        },
        round_state_exchange={
            "exchange_name": "none",
            "required_client_metric_keys": [],
            "custom_exchange_required": False,
        },
        report_tags=["baseline", "fedavg", "pseudo_label"],
    )


def _default_round_runtime_config(
    *,
    adapter_family_name: str = "diagonal_scale",
    aggregation_backend_name: str = "fedavg",
    classifier_head_bootstrap_logit_scale: float = 8.0,
    lora_classifier: FederatedLoraClassifierRuntimeConfig | None = None,
) -> FederatedRoundRuntimeConfig:
    return FederatedRoundRuntimeConfig(
        adapter_family_name=adapter_family_name,
        aggregation_backend_name=aggregation_backend_name,
        classifier_head_bootstrap_logit_scale=classifier_head_bootstrap_logit_scale,
        lora_classifier=lora_classifier,
    )


def _lora_runtime_config() -> FederatedLoraClassifierRuntimeConfig:
    return FederatedLoraClassifierRuntimeConfig(
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


def _lora_objective_extras() -> dict[str, str | int | float | bool]:
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
        "lora_classifier.delta_format": "inline_delta",
        "lora_classifier.artifact_ref_prefix": "agent-local://lora_classifier",
        "lora_classifier.text_metadata_keys": (
            "strong_text,training_text,raw_text,text,weak_text"
        ),
        "lora_classifier.label_schema": "anxiety,depression,normal,suicidal",
    }


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


def test_federated_ssl_simulation_runtime_uses_methods_descriptor() -> None:
    descriptor = resolve_federated_ssl_method_descriptor("fedavg_pseudo_label")

    assert descriptor.implementation_status == "active_runtime"
    assert descriptor.client_trainer_name == "local_training_service"
    assert descriptor.pseudo_labeler_name == "ssl_pseudo_label_selection_hook"
    assert descriptor.server_aggregator_name == "round_runtime_aggregation_backend"
    assert descriptor.requires_custom_client_runtime is False
    assert descriptor.requires_custom_server_runtime is False
    runtime = build_federated_ssl_simulation_runtime("fedavg_pseudo_label")
    assert runtime.descriptor is descriptor

    with pytest.raises(NotImplementedError, match="descriptor is not wired yet"):
        build_federated_ssl_simulation_runtime("paper_method_candidate")


def test_federated_ssl_runtime_uses_method_training_row_source() -> None:
    runtime = build_federated_ssl_simulation_runtime("fedavg_pseudo_label")
    labeled_row = _row("l1", "labeled", "normal")
    unlabeled_row = _row("u1", "unlabeled", "anxiety")

    selected_rows = runtime.select_training_rows(
        shard=FederatedClientShard(
            client_id="agent_001",
            rows=[labeled_row, unlabeled_row],
            labeled_rows=[labeled_row],
            unlabeled_rows=[unlabeled_row],
            client_pool_split_enforced=True,
        )
    )

    assert selected_rows == [unlabeled_row]


def test_simulation_server_runtime_wires_method_descriptor(tmp_path: Path) -> None:
    descriptor = resolve_federated_ssl_method_descriptor("fedavg_pseudo_label")

    runtime = SimulationServerRuntime.build(
        output_dir=tmp_path,
        round_runtime_config=_default_round_runtime_config(),
        prototype_build_strategy=SinglePrototypeBuildStrategy(),
        method_descriptor=descriptor,
    )

    assert runtime.lifecycle_service.method_descriptor is descriptor


def test_federated_ssl_runtime_rejects_method_config_descriptor_drift() -> None:
    ssl_method_config = _default_ssl_method_config()
    ssl_method_config.client_step["task_type"] = "supervised_mix_local_training"

    with pytest.raises(ValueError, match="ssl_method.client_step.*task_type"):
        _build_validated_ssl_runtime(ssl_method_config)


def test_federated_ssl_runtime_rejects_round_state_metric_key_drift() -> None:
    ssl_method_config = _default_ssl_method_config()
    ssl_method_config.round_state_exchange["required_client_metric_keys"] = [
        "client_entropy"
    ]

    with pytest.raises(
        ValueError,
        match="ssl_method.round_state_exchange.*required_client_metric_keys",
    ):
        _build_validated_ssl_runtime(ssl_method_config)


def test_run_simulation_request_rejects_training_task_type_descriptor_drift(
    tmp_path,
) -> None:
    request = SimulationRunRequest(
        train_rows=[
            _row("a1", "panic panic", "anxiety"),
            _row("n1", "calm calm", "normal"),
        ],
        validation_rows=[_row("va", "panic panic", "anxiety")],
        output_dir=tmp_path / "task_type_mismatch",
        client_count=2,
        rounds=0,
        bootstrap_ratio=0.5,
        seed=7,
        embedding_spec=EmbeddingAdapterSpec(
            backend="hash_debug",
            model_id="hash_debug",
            revision="sim",
            hash_dim=32,
        ),
        model_id="tracemind-embed-sim",
        training_scope="adapter_only",
        round_runtime_config=_default_round_runtime_config(),
        prototype_build_strategy=SinglePrototypeBuildStrategy(),
        shard_policy=_default_shard_policy(),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
            task_type=TrainingTaskType.FEEDBACK_SUPERVISED,
        ),
        validation_config=_default_validation_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
        ),
        prototype_rebuild_config=_default_prototype_rebuild_config(),
        diagnostics_config=_default_diagnostics_config(),
        ssl_method_config=_default_ssl_method_config(),
        client_pool_split_config=_default_client_pool_split_config(),
    )

    with pytest.raises(ValueError, match="training_task_config.task_type"):
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


def test_build_initial_shared_state_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="Unsupported simulation adapter family"):
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
    drifted_lora_runtime = FederatedLoraClassifierRuntimeConfig(
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
    request = SimulationRunRequest(
        train_rows=[
            _row("a1", "panic panic", "anxiety"),
            _row("n1", "calm calm", "normal"),
        ],
        validation_rows=[_row("va", "panic panic", "anxiety")],
        output_dir=tmp_path / "lora_drift",
        client_count=2,
        rounds=0,
        bootstrap_ratio=0.5,
        seed=7,
        embedding_spec=EmbeddingAdapterSpec(
            backend="hash_debug",
            model_id="hash_debug",
            revision="sim",
            hash_dim=32,
        ),
        model_id="mxbai-lora-classifier",
        training_scope="adapter_only",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=drifted_lora_runtime,
        ),
        prototype_build_strategy=SinglePrototypeBuildStrategy(),
        shard_policy=_default_shard_policy(),
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
        ),
        prototype_rebuild_config=_default_prototype_rebuild_config(),
        diagnostics_config=_default_diagnostics_config(),
        ssl_method_config=_default_ssl_method_config(),
        client_pool_split_config=_default_client_pool_split_config(),
    )

    with pytest.raises(ValueError, match="LoRA-classifier.*training_task.objective"):
        run_simulation_request(request)


def test_run_simulation_request_bootstraps_lora_classifier_profile(
    tmp_path,
) -> None:
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
    request = SimulationRunRequest(
        train_rows=train_rows,
        validation_rows=validation_rows,
        output_dir=tmp_path / "lora_simulation_request",
        client_count=2,
        rounds=0,
        bootstrap_ratio=0.5,
        seed=7,
        embedding_spec=EmbeddingAdapterSpec(
            backend="hash_debug",
            model_id="hash_debug",
            revision="sim",
            hash_dim=32,
        ),
        model_id="mxbai-lora-classifier",
        training_scope="adapter_only",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=_lora_runtime_config(),
        ),
        prototype_build_strategy=SinglePrototypeBuildStrategy(),
        shard_policy=_default_shard_policy(),
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
        ),
        prototype_rebuild_config=_default_prototype_rebuild_config(),
        diagnostics_config=_default_diagnostics_config(),
        ssl_method_config=_default_ssl_method_config(),
        client_pool_split_config=_default_client_pool_split_config(),
    )

    result = run_simulation_request(request)

    assert result.initial_model_revision == "sim_rev_0000"
    assert result.rounds == ()
    assert result.final_validation == result.initial_validation


def test_run_simulation_request_completes_lora_classifier_inline_delta_rounds(
    tmp_path,
) -> None:
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
    request = SimulationRunRequest(
        train_rows=train_rows,
        validation_rows=validation_rows,
        output_dir=output_dir,
        client_count=4,
        rounds=2,
        bootstrap_ratio=1 / 3,
        seed=7,
        embedding_spec=EmbeddingAdapterSpec(
            backend="hash_debug",
            model_id="hash_debug",
            revision="sim",
            hash_dim=32,
        ),
        model_id="mxbai-lora-classifier",
        training_scope="adapter_only",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=_lora_runtime_config(),
        ),
        prototype_build_strategy=SinglePrototypeBuildStrategy(),
        shard_policy=_default_shard_policy(),
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
        ),
        prototype_rebuild_config=_default_prototype_rebuild_config(),
        diagnostics_config=_default_diagnostics_config(),
        ssl_method_config=_default_ssl_method_config(),
        client_pool_split_config=_default_client_pool_split_config(),
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
    assert update_payload["delta_format"] == "inline_delta"
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


def test_run_simulation_request_rejects_local_round_family_mismatch(
    tmp_path,
) -> None:
    request = SimulationRunRequest(
        train_rows=[
            _row("a1", "panic panic", "anxiety"),
            _row("n1", "calm calm", "normal"),
        ],
        validation_rows=[_row("va", "panic panic", "anxiety")],
        output_dir=tmp_path / "mismatch_simulation_request",
        client_count=2,
        rounds=0,
        bootstrap_ratio=0.5,
        seed=7,
        embedding_spec=EmbeddingAdapterSpec(
            backend="hash_debug",
            model_id="hash_debug",
            revision="sim",
            hash_dim=32,
        ),
        model_id="mxbai-lora-classifier",
        training_scope="adapter_only",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="lora_classifier",
            lora_classifier=_lora_runtime_config(),
        ),
        prototype_build_strategy=SinglePrototypeBuildStrategy(),
        shard_policy=_default_shard_policy(),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
        ),
        validation_config=_default_validation_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
        ),
        prototype_rebuild_config=_default_prototype_rebuild_config(),
        diagnostics_config=_default_diagnostics_config(),
        ssl_method_config=_default_ssl_method_config(),
    )

    with pytest.raises(ValueError, match="local_update_profile.*round_runtime_profile"):
        run_simulation_request(request)


def test_run_simulation_completes_one_round_with_small_fixture(tmp_path) -> None:
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

    result = run_simulation(
        train_rows=train_rows,
        validation_rows=validation_rows,
        output_dir=tmp_path / "simulation",
        client_count=4,
        rounds=1,
        bootstrap_ratio=1 / 3,
        seed=7,
        embedding_spec=EmbeddingAdapterSpec(
            backend="hash_debug",
            model_id="hash_debug",
            revision="sim",
            hash_dim=32,
        ),
        model_id="tracemind-embed-sim",
        training_scope="adapter_only",
        round_runtime_config=_default_round_runtime_config(),
        prototype_build_strategy=SinglePrototypeBuildStrategy(),
        shard_policy=_default_shard_policy(),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
        ),
        validation_config=_default_validation_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
        ),
        prototype_rebuild_config=_default_prototype_rebuild_config(),
        diagnostics_config=_default_diagnostics_config(),
        ssl_method_config=_default_ssl_method_config(),
        client_pool_split_config=_default_client_pool_split_config(),
        report_config=_default_report_config(),
    )

    assert result.rounds
    assert result.rounds[0].update_count > 0
    assert result.rounds[0].model_revision == "sim_rev_0001"
    assert result.rounds[0].prototype_version == "proto_sim_0001"
    dump_paths = sorted(
        (tmp_path / "simulation" / "agents").glob(
            "*/selection_dumps/round_0001.summary.json"
        )
    )
    assert dump_paths
    summary = json.loads(dump_paths[0].read_text(encoding="utf-8"))
    assert "stage_counts" in summary
    candidate_paths = sorted(
        (tmp_path / "simulation" / "agents").glob(
            "*/selection_dumps/round_0001.candidates.jsonl"
        )
    )
    assert candidate_paths
    first_line = candidate_paths[0].read_text(encoding="utf-8").splitlines()[0]
    first_candidate = json.loads(first_line)
    assert "selection_stage" in first_candidate
    assert "threshold_accepted" in first_candidate
    assert result.report_path is not None
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert report["track"] == "fl_ssl_main_comparison"
    assert report["table_role"] == "main_comparison"
    assert report["must_not_merge_with"] == ["central_ssl_control"]
    assert report["protocol"]["round_budget"] == 1
    assert report["protocol"]["ssl_method"]["name"] == "fedavg_pseudo_label"
    assert report["protocol"]["ssl_method"]["method_role"] == "baseline"
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
        "negative_log_likelihood_from_score_distribution"
    )
    assert report["metrics"]["final_validation"]["score_distribution_kind"] == (
        "softmax_raw_scores_temperature_1.0"
    )
    assert report["metrics"]["round_progression"]["round_count"] == 1
    assert (
        report["metrics"]["round_progression"]["early_stop_candidate"]["status"]
        == "insufficient_rounds"
    )
    assert report["rounds"][0]["round_index"] == 1
    assert "accepted_ratio" in report["rounds"][0]["clients"][0]
    assert "delta_l2_norm" in report["rounds"][0]["clients"][0]
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


def test_run_simulation_request_preserves_typed_boundary(tmp_path) -> None:
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
    request = SimulationRunRequest(
        train_rows=train_rows,
        validation_rows=validation_rows,
        output_dir=tmp_path / "simulation_request",
        client_count=2,
        rounds=0,
        bootstrap_ratio=0.5,
        seed=7,
        embedding_spec=EmbeddingAdapterSpec(
            backend="hash_debug",
            model_id="hash_debug",
            revision="sim",
            hash_dim=32,
        ),
        model_id="tracemind-embed-sim",
        training_scope="adapter_only",
        round_runtime_config=_default_round_runtime_config(),
        prototype_build_strategy=SinglePrototypeBuildStrategy(),
        shard_policy=_default_shard_policy(),
        training_task_config=_default_training_task_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
            max_examples=4,
            gradient_clip_norm=1.0,
        ),
        validation_config=_default_validation_config(
            confidence_threshold=0.0,
            margin_threshold=0.0,
        ),
        prototype_rebuild_config=_default_prototype_rebuild_config(),
        diagnostics_config=_default_diagnostics_config(),
        ssl_method_config=_default_ssl_method_config(),
        client_pool_split_config=_default_client_pool_split_config(),
    )

    result = run_simulation_request(request)

    assert result.initial_model_revision == "sim_rev_0000"
    assert result.initial_prototype_version == "proto_sim_0000"
    assert result.rounds == ()
    assert result.final_validation == result.initial_validation
    assert len(result.client_evaluations) == 2


def test_run_simulation_accepts_hydra_style_detail_configs(tmp_path) -> None:
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

    result = run_simulation(
        train_rows=train_rows,
        validation_rows=validation_rows,
        output_dir=tmp_path / "simulation",
        client_count=4,
        rounds=1,
        bootstrap_ratio=1 / 3,
        seed=7,
        embedding_spec=EmbeddingAdapterSpec(
            backend="hash_debug",
            model_id="hash_debug",
            revision="sim",
            hash_dim=32,
        ),
        model_id="tracemind-embed-sim",
        training_scope="adapter_only",
        round_runtime_config=_default_round_runtime_config(),
        prototype_build_strategy=SinglePrototypeBuildStrategy(),
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
        prototype_rebuild_config=_default_prototype_rebuild_config(),
        diagnostics_config=_default_diagnostics_config(),
        ssl_method_config=_default_ssl_method_config(),
        client_pool_split_config=_default_client_pool_split_config(),
    )

    assert result.rounds
    assert result.rounds[0].update_count > 0


def test_evaluate_rows_respects_acceptance_policy_for_acceptance_ratio() -> None:
    rows = [_row("q1", "panic panic", "anxiety")]
    adapter = _StaticEmbeddingAdapter({"panic panic": [0.6, 0.4]})
    adapter_state = VectorAdapterState.identity(
        model_id="hash_debug",
        model_revision="main",
        training_scope="adapter_only",
        embedding_dim=2,
        updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )

    objective_config = TrainingObjectiveConfig.from_mapping(
        {
            "training_backend_name": "diagonal_scale_heuristic",
            "example_generation_backend_name": "prototype_rescore",
            "evidence_backend_name": "prototype_similarity_evidence",
            "scorer_backend_name": "prototype_similarity",
            "score_policy_name": "max_cosine",
            "pseudo_label_algorithm_name": "top1_confidence_only",
            "acceptance_policy_name": "top1_confidence_only",
            "privacy_guard_name": "noop",
        }
    )
    result = evaluate_rows(
        rows=rows,
        adapter=adapter,
        adapter_state=adapter_state,
        prototype_pack=_pack_payload(),
        model_id="hash_debug",
        scoring_service=build_federated_scoring_service(
            objective_config=objective_config,
            similarity_name="cosine",
        ),
        confidence_threshold=0.9,
        margin_threshold=0.0,
        objective_config=objective_config,
    )

    assert result.top1_accuracy == 1.0
    assert result.accepted_ratio == 0.0
    assert result.loss >= 0.0
    assert result.loss_kind == "negative_log_likelihood_from_score_distribution"
    assert result.accuracy_top_1 == 1.0
    assert result.correct_top_1 == 1
    assert result.macro_f1 == 1.0
    assert result.weighted_f1 == 1.0
    assert result.expected_calibration_error >= 0.0
    assert result.score_distribution_kind == "softmax_raw_scores_temperature_1.0"
    assert result.classification_report["loss"] == result.loss
    assert result.confusion_matrix == {"anxiety": {"anxiety": 1}}
    assert result.per_label["anxiety"]["f1"] == 1.0


def test_build_training_examples_requires_multiview_fields() -> None:
    adapter = _StaticEmbeddingAdapter({"panic panic": [1.0, 0.0]})
    adapter_state = VectorAdapterState.identity(
        model_id="hash_debug",
        model_revision="main",
        training_scope="adapter_only",
        embedding_dim=2,
        updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )

    with pytest.raises(
        ValueError,
        match="requires each row to include both weak_text and strong_text",
    ):
        objective_config = TrainingObjectiveConfig.from_mapping(
            {
                "training_backend_name": "diagonal_scale_heuristic",
                "example_generation_backend_name": "weak_strong_pair",
                "evidence_backend_name": "prototype_similarity_evidence",
                "scorer_backend_name": "prototype_similarity",
                "score_policy_name": "max_cosine",
                "pseudo_label_algorithm_name": "top1_margin_threshold",
                "acceptance_policy_name": "top1_margin_threshold",
                "privacy_guard_name": "noop",
            }
        )
        build_training_examples(
            rows=[_row("q1", "panic panic", "anxiety")],
            adapter=adapter,
            adapter_state=adapter_state,
            prototype_pack=_pack_payload(),
            model_id="hash_debug",
            scoring_service=build_federated_scoring_service(
                objective_config=objective_config,
                similarity_name="cosine",
            ),
            objective_config=objective_config,
        )


def test_build_training_examples_supports_multiview_row_fields_when_present() -> None:
    adapter = _StaticEmbeddingAdapter(
        {
            "panic weak": [1.0, 0.0],
            "panic strong": [0.8, 0.2],
        }
    )
    adapter_state = VectorAdapterState.identity(
        model_id="hash_debug",
        model_revision="main",
        training_scope="adapter_only",
        embedding_dim=2,
        updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )
    row = _row("q1", "panic panic", "anxiety")
    row["weak_text"] = "panic weak"
    row["strong_text"] = "panic strong"

    objective_config = TrainingObjectiveConfig.from_mapping(
        {
            "training_backend_name": "diagonal_scale_heuristic",
            "example_generation_backend_name": "weak_strong_pair",
            "evidence_backend_name": "prototype_similarity_evidence",
            "scorer_backend_name": "prototype_similarity",
            "score_policy_name": "max_cosine",
            "pseudo_label_algorithm_name": "top1_margin_threshold",
            "acceptance_policy_name": "top1_margin_threshold",
            "privacy_guard_name": "noop",
        }
    )
    examples = build_training_examples(
        rows=[row],
        adapter=adapter,
        adapter_state=adapter_state,
        prototype_pack=_pack_payload(),
        model_id="hash_debug",
        scoring_service=build_federated_scoring_service(
            objective_config=objective_config,
            similarity_name="cosine",
        ),
        objective_config=objective_config,
    )

    assert len(examples) == 1
    assert examples[0].view_kind == "weak_strong_pair"
    assert examples[0].weak_embedding == [1.0, 0.0]
    assert examples[0].strong_embedding == pytest.approx(
        [0.9701425001453318, 0.24253562503633294]
    )


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
