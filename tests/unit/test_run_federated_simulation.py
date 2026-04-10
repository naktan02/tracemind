"""run_federated_simulation 스크립트 unit tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agent.src.services.inference.scoring_service import ScoringService
from main_server.src.services.rounds.models import RoundTaskConfig
from scripts.experiments.federated_simulation import (
    FederatedDiagnosticsConfig,
    FederatedPrototypeRebuildConfig,
    FederatedRoundRuntimeConfig,
    FederatedShardPolicyConfig,
    FederatedTrainingTaskConfig,
    FederatedValidationConfig,
)
from scripts.experiments.federated_simulation.evaluation import (
    build_training_examples,
    evaluate_rows,
)
from scripts.experiments.federated_simulation.task_config import (
    build_round_open_request,
)
from scripts.experiments.run_federated_simulation import (
    run_simulation,
    split_rows_for_federation,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)
from shared.src.contracts.adapter_contracts import VectorAdapterState
from shared.src.domain.value_objects import EmbeddingAdapterSpec
from shared.src.services.prototypes.build_strategies import (
    KMeansPrototypeBuildStrategy,
    SinglePrototypeBuildStrategy,
)


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
    scorer_backend_name: str = "prototype_similarity",
    score_policy_name: str = "max_cosine",
    score_top_k: int | None = None,
) -> FederatedTrainingTaskConfig:
    return FederatedTrainingTaskConfig(
        local_epochs=1,
        batch_size=16,
        learning_rate=1e-4,
        max_steps=50,
        min_required_examples=1,
        gradient_clip_norm=gradient_clip_norm,
        objective_config=TrainingObjectiveConfig.from_mapping(
            {
                "training_backend_name": "diagonal_scale_heuristic",
                "confidence_threshold": confidence_threshold,
                "margin_threshold": margin_threshold,
                "example_generation_backend_name": "prototype_rescore",
                "scorer_backend_name": scorer_backend_name,
                "score_policy_name": score_policy_name,
                **(
                    {}
                    if score_top_k is None
                    else {"score_top_k": score_top_k}
                ),
                "acceptance_policy_name": "top1_margin_threshold",
                "privacy_guard_name": "diagonal_scale_clip_only",
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


def _default_round_runtime_config(
    *,
    adapter_family_name: str = "diagonal_scale",
    aggregation_backend_name: str = "fedavg",
    classifier_head_bootstrap_logit_scale: float = 8.0,
) -> FederatedRoundRuntimeConfig:
    return FederatedRoundRuntimeConfig(
        adapter_family_name=adapter_family_name,
        aggregation_backend_name=aggregation_backend_name,
        classifier_head_bootstrap_logit_scale=classifier_head_bootstrap_logit_scale,
    )


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


def test_federated_training_task_config_reuses_round_task_config() -> None:
    training_task_config = _default_training_task_config(
        confidence_threshold=0.6,
        margin_threshold=0.02,
        max_examples=8,
        gradient_clip_norm=0.5,
    )

    request = build_round_open_request(
        active_manifest=ModelManifest(
            schema_version="model_manifest.v1",
            model_id="tracemind-embed",
            model_revision="rev_000",
            published_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
            artifact_kind="shared_adapter_state",
            artifact_ref="/tmp/rev_000.json",
            prototype_version="proto_000",
            training_scope="adapter_only",
            training_enabled=True,
            compatible_task_types=("pseudo_label_self_training",),
        ),
        round_id="round_0001",
        training_task_config=training_task_config,
    )

    assert isinstance(training_task_config, RoundTaskConfig)
    assert request.local_epochs == training_task_config.local_epochs
    assert request.batch_size == training_task_config.batch_size
    assert request.learning_rate == training_task_config.learning_rate
    assert request.max_steps == training_task_config.max_steps
    assert request.objective_config is training_task_config.objective_config
    assert request.selection_policy is training_task_config.selection_policy


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
    )

    assert result.rounds
    assert result.rounds[0].update_count > 0


def test_evaluate_rows_uses_evidence_backend_for_acceptance_ratio() -> None:
    rows = [_row("q1", "panic panic", "anxiety")]
    adapter = _StaticEmbeddingAdapter({"panic panic": [1.0, 0.0]})
    adapter_state = VectorAdapterState.identity(
        model_id="hash_debug",
        model_revision="main",
        training_scope="adapter_only",
        embedding_dim=2,
        updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )

    result = evaluate_rows(
        rows=rows,
        adapter=adapter,
        adapter_state=adapter_state,
        prototype_pack=_pack_payload(),
        model_id="hash_debug",
        scoring_service=ScoringService(),
        confidence_threshold=0.8,
        margin_threshold=0.0,
        objective_config=TrainingObjectiveConfig.from_mapping(
            {
                "training_backend_name": "diagonal_scale_heuristic",
                "example_generation_backend_name": "prototype_rescore",
                "evidence_backend_name": "fixmatch_weak_view_evidence",
                "scorer_backend_name": "prototype_similarity",
                "score_policy_name": "max_cosine",
                "acceptance_policy_name": "top1_confidence_only",
                "privacy_guard_name": "noop",
            }
        ),
    )

    assert result.top1_accuracy == 1.0
    assert result.accepted_ratio == 0.0


def test_build_training_examples_requires_multiview_fields_for_weak_strong_backend() -> None:
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
        build_training_examples(
            rows=[_row("q1", "panic panic", "anxiety")],
            adapter=adapter,
            adapter_state=adapter_state,
            prototype_pack=_pack_payload(),
            model_id="hash_debug",
            scoring_service=ScoringService(),
            objective_config=TrainingObjectiveConfig.from_mapping(
                {
                    "training_backend_name": "diagonal_scale_heuristic",
                    "example_generation_backend_name": "weak_strong_pair",
                    "evidence_backend_name": "prototype_similarity_evidence",
                    "scorer_backend_name": "prototype_similarity",
                    "score_policy_name": "max_cosine",
                    "acceptance_policy_name": "top1_margin_threshold",
                    "privacy_guard_name": "noop",
                }
            ),
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

    examples = build_training_examples(
        rows=[row],
        adapter=adapter,
        adapter_state=adapter_state,
        prototype_pack=_pack_payload(),
        model_id="hash_debug",
        scoring_service=ScoringService(),
        objective_config=TrainingObjectiveConfig.from_mapping(
            {
                "training_backend_name": "diagonal_scale_heuristic",
                "example_generation_backend_name": "weak_strong_pair",
                "evidence_backend_name": "prototype_similarity_evidence",
                "scorer_backend_name": "prototype_similarity",
                "score_policy_name": "max_cosine",
                "acceptance_policy_name": "top1_margin_threshold",
                "privacy_guard_name": "noop",
            }
        ),
    )

    assert len(examples) == 1
    assert examples[0].view_kind == "weak_strong_pair"
    assert examples[0].weak_embedding == [1.0, 0.0]
    assert examples[0].strong_embedding == pytest.approx(
        [0.9701425001453318, 0.24253562503633294]
    )


def test_run_simulation_supports_classifier_head_fixmatch_path(tmp_path) -> None:
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
    for row in (*train_rows, *validation_rows):
        row["weak_text"] = str(row["text"])
        row["strong_text"] = str(row["text"])

    result = run_simulation(
        train_rows=train_rows,
        validation_rows=validation_rows,
        output_dir=tmp_path / "simulation_fixmatch",
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
        training_scope="head_only",
        round_runtime_config=_default_round_runtime_config(
            adapter_family_name="classifier_head",
            aggregation_backend_name="fedavg",
        ),
        prototype_build_strategy=SinglePrototypeBuildStrategy(),
        shard_policy=_default_shard_policy(),
        training_task_config=FederatedTrainingTaskConfig(
            local_epochs=1,
            batch_size=16,
            learning_rate=1e-2,
            max_steps=1,
            min_required_examples=1,
            gradient_clip_norm=1.0,
            objective_config=TrainingObjectiveConfig.from_mapping(
                {
                    "algorithm_profile_name": "fixmatch_v1",
                    "training_backend_name": "classifier_head_fixmatch_consistency",
                    "confidence_threshold": 0.95,
                    "margin_threshold": 0.0,
                    "example_generation_backend_name": "weak_strong_pair",
                    "evidence_backend_name": "fixmatch_weak_view_evidence",
                    "scorer_backend_name": "classifier_head_logits",
                    "acceptance_policy_name": "top1_confidence_only",
                    "privacy_guard_name": "classifier_head_clip_only",
                    "training_backend.consistency_loss_weight": 1.0,
                    "training_backend.step_scale_multiplier": 1.0,
                    "training_backend.bias_learning_rate_multiplier": 1.0,
                }
            ),
            selection_policy=TrainingSelectionPolicy(max_examples=8),
        ),
        validation_config=_default_validation_config(
            confidence_threshold=0.95,
            margin_threshold=0.0,
            scorer_backend_name="classifier_head_logits",
            score_policy_name=None,
        ),
        prototype_rebuild_config=_default_prototype_rebuild_config(),
        diagnostics_config=_default_diagnostics_config(),
    )

    assert result.rounds
    assert result.rounds[0].update_count > 0


def test_run_simulation_rejects_classifier_head_with_multi_prototype_builder(
    tmp_path,
) -> None:
    train_rows = [
        _row("a1", "panic panic", "anxiety"),
        _row("a2", "panic panic", "anxiety"),
        _row("n1", "calm calm", "normal"),
        _row("n2", "calm calm", "normal"),
        _row("a3", "panic panic", "anxiety"),
        _row("n3", "calm calm", "normal"),
    ]
    validation_rows = [
        _row("va1", "panic panic", "anxiety"),
        _row("vn1", "calm calm", "normal"),
    ]
    for row in (*train_rows, *validation_rows):
        row["weak_text"] = str(row["text"])
        row["strong_text"] = str(row["text"])

    with pytest.raises(ValueError, match="prototype_builder=single"):
        run_simulation(
            train_rows=train_rows,
            validation_rows=validation_rows,
            output_dir=tmp_path / "simulation_fixmatch_kmeans",
            client_count=2,
            rounds=1,
            bootstrap_ratio=0.5,
            seed=7,
            embedding_spec=EmbeddingAdapterSpec(
                backend="hash_debug",
                model_id="hash_debug",
                revision="sim",
                hash_dim=32,
            ),
            model_id="tracemind-embed-sim",
            training_scope="head_only",
            round_runtime_config=_default_round_runtime_config(
                adapter_family_name="classifier_head",
                aggregation_backend_name="fedavg",
            ),
            prototype_build_strategy=KMeansPrototypeBuildStrategy(candidate_ks=(2,)),
            shard_policy=_default_shard_policy(),
            training_task_config=_default_training_task_config(
                confidence_threshold=0.95,
                margin_threshold=0.0,
                max_examples=8,
                gradient_clip_norm=1.0,
                scorer_backend_name="classifier_head_logits",
                score_policy_name=None,
            ),
            validation_config=_default_validation_config(
                confidence_threshold=0.95,
                margin_threshold=0.0,
                scorer_backend_name="classifier_head_logits",
                score_policy_name=None,
            ),
            prototype_rebuild_config=_default_prototype_rebuild_config(),
            diagnostics_config=_default_diagnostics_config(),
        )
