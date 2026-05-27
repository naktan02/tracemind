from __future__ import annotations

from methods.federated_ssl.runtime_fallbacks import (
    LEGACY_RUNTIME_FALLBACK_TRAINING_BACKEND_EXTRAS,
    PSEUDO_LABEL_SELF_TRAINING_V1_RUNTIME_FALLBACK,
    RUNTIME_FALLBACK_TRAINING_PROFILE,
    RUNTIME_FALLBACK_TRAINING_TASK_DEFAULTS,
    build_runtime_fallback_secure_aggregation_config,
    build_runtime_fallback_training_objective_config,
    build_runtime_fallback_training_selection_policy,
)


def test_runtime_fallback_profile_points_to_versioned_bundle() -> None:
    assert (
        RUNTIME_FALLBACK_TRAINING_PROFILE
        is PSEUDO_LABEL_SELF_TRAINING_V1_RUNTIME_FALLBACK
    )
    assert (
        RUNTIME_FALLBACK_TRAINING_PROFILE.profile_name
        == "pseudo_label_self_training.v1"
    )


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
    assert config.extras == dict(LEGACY_RUNTIME_FALLBACK_TRAINING_BACKEND_EXTRAS)


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
