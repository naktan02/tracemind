from __future__ import annotations

from methods.adaptation.peft_text_encoder.config import PEFT_ENCODER_DELTA_FORMAT_INLINE
from methods.federated_ssl.runtime_fallbacks import (
    FIXMATCH_FEDAVG_V1_RUNTIME_FALLBACK,
    FIXMATCH_QUERY_SSL_ALGORITHM_NAME,
    FIXMATCH_QUERY_SSL_METHOD_NAME,
    FIXMATCH_QUERY_SSL_P_CUTOFF,
    FIXMATCH_QUERY_SSL_STRONG_VIEW_POLICY,
    FIXMATCH_QUERY_SSL_TEMPERATURE,
    RUNTIME_FALLBACK_TRAINING_PROFILE,
    RUNTIME_FALLBACK_TRAINING_TASK_DEFAULTS,
    build_runtime_fallback_secure_aggregation_config,
    build_runtime_fallback_training_objective_config,
    build_runtime_fallback_training_selection_policy,
    resolve_runtime_example_generation_backend_name,
)
from shared.src.contracts.training_example_backends import (
    WEAK_STRONG_PAIR_EXAMPLE_BACKEND,
)


def test_runtime_fallback_profile_points_to_versioned_bundle() -> None:
    assert RUNTIME_FALLBACK_TRAINING_PROFILE is FIXMATCH_FEDAVG_V1_RUNTIME_FALLBACK
    assert RUNTIME_FALLBACK_TRAINING_PROFILE.profile_name == "fixmatch_fedavg.v1"


def test_runtime_fallback_objective_builder_uses_runtime_fallback_profile() -> None:
    config = build_runtime_fallback_training_objective_config()

    assert (
        config.training_backend_name
        == RUNTIME_FALLBACK_TRAINING_PROFILE.training_backend_name
    )
    assert (
        config.algorithm_profile_name
        == RUNTIME_FALLBACK_TRAINING_PROFILE.algorithm_profile_name
    )
    assert (
        config.confidence_threshold
        == RUNTIME_FALLBACK_TRAINING_PROFILE.confidence_threshold
    )
    assert config.margin_threshold == RUNTIME_FALLBACK_TRAINING_PROFILE.margin_threshold
    assert (
        config.example_generation_backend_name
        == RUNTIME_FALLBACK_TRAINING_PROFILE.example_generation_backend_name
    )
    assert (
        config.evidence_backend_name
        == RUNTIME_FALLBACK_TRAINING_PROFILE.evidence_backend_name
    )
    assert (
        config.scorer_backend_name
        == RUNTIME_FALLBACK_TRAINING_PROFILE.scorer_backend_name
    )
    assert (
        config.privacy_guard_name
        == RUNTIME_FALLBACK_TRAINING_PROFILE.privacy_guard_name
    )
    assert (
        config.pseudo_label_algorithm_name
        == RUNTIME_FALLBACK_TRAINING_PROFILE.pseudo_label_algorithm_name
    )
    assert (
        config.acceptance_policy_name
        == RUNTIME_FALLBACK_TRAINING_PROFILE.acceptance_policy_name
    )
    assert config.extras == {
        "query_ssl.method_name": FIXMATCH_QUERY_SSL_METHOD_NAME,
        "query_ssl.algorithm_name": FIXMATCH_QUERY_SSL_ALGORITHM_NAME,
        "query_ssl.strong_view_policy": FIXMATCH_QUERY_SSL_STRONG_VIEW_POLICY,
        "query_ssl.unlabeled_batch_size": 12,
        "query_ssl.temperature": FIXMATCH_QUERY_SSL_TEMPERATURE,
        "query_ssl.p_cutoff": FIXMATCH_QUERY_SSL_P_CUTOFF,
        "query_ssl.hard_label": True,
        "query_ssl.lambda_u": 1.0,
        "query_ssl.supervised_loss_weight": 1.0,
        "peft_classifier.delta_format": PEFT_ENCODER_DELTA_FORMAT_INLINE,
    }
    assert config.example_generation_backend_name == WEAK_STRONG_PAIR_EXAMPLE_BACKEND


def test_runtime_fallback_objective_builder_accepts_overrides() -> None:
    config = build_runtime_fallback_training_objective_config(
        overrides={
            "confidence_threshold": 0.75,
            "score_top_k": 3,
            "training_backend.max_abs_delta": 0.02,
        }
    )

    assert config.confidence_threshold == 0.75
    assert config.score_top_k == 3
    assert config.extras["training_backend.max_abs_delta"] == 0.02


def test_runtime_fallback_selection_policy_builder_uses_profile() -> None:
    policy = build_runtime_fallback_training_selection_policy()

    assert policy.max_examples == RUNTIME_FALLBACK_TRAINING_PROFILE.max_examples


def test_runtime_fallback_secure_aggregation_builder_starts_disabled() -> None:
    config = build_runtime_fallback_secure_aggregation_config()

    assert config.required is False


def test_runtime_fallback_profile_exposes_round_task_runtime_defaults() -> None:
    assert RUNTIME_FALLBACK_TRAINING_PROFILE.local_epochs == 1
    assert RUNTIME_FALLBACK_TRAINING_PROFILE.batch_size == 12
    assert RUNTIME_FALLBACK_TRAINING_PROFILE.learning_rate == 1e-4
    assert RUNTIME_FALLBACK_TRAINING_PROFILE.max_steps == 50
    assert (
        RUNTIME_FALLBACK_TRAINING_PROFILE.task_runtime_defaults
        is RUNTIME_FALLBACK_TRAINING_TASK_DEFAULTS
    )


def test_runtime_fallback_resolves_example_generation_backend_name() -> None:
    assert (
        resolve_runtime_example_generation_backend_name(None)
        == WEAK_STRONG_PAIR_EXAMPLE_BACKEND
    )
    assert (
        resolve_runtime_example_generation_backend_name(
            type("Objective", (), {"example_generation_backend_name": ""})()
        )
        == WEAK_STRONG_PAIR_EXAMPLE_BACKEND
    )
    assert (
        resolve_runtime_example_generation_backend_name(
            type(
                "Objective",
                (),
                {"example_generation_backend_name": "prototype_rescore"},
            )()
        )
        == "prototype_rescore"
    )
