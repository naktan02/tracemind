from __future__ import annotations

from methods.adaptation.peft_text_encoder.config import PEFT_ENCODER_DELTA_FORMAT_INLINE
from methods.federated_ssl.runtime_fallbacks import (
    FIXMATCH_FEDAVG_V1_RUNTIME_FALLBACK,
    FIXMATCH_QUERY_SSL_METHOD_NAME,
    FIXMATCH_QUERY_SSL_STRONG_VIEW_POLICY,
    QUERY_SSL_METHOD_OBJECTIVE_DEFAULTS,
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
    query_ssl_defaults = QUERY_SSL_METHOD_OBJECTIVE_DEFAULTS[
        FIXMATCH_QUERY_SSL_METHOD_NAME
    ]

    assert (
        config.training_backend_name
        == RUNTIME_FALLBACK_TRAINING_PROFILE.training_backend_name
    )
    assert (
        config.algorithm_profile_name
        == RUNTIME_FALLBACK_TRAINING_PROFILE.algorithm_profile_name
    )
    assert (
        config.example_generation_backend_name
        == RUNTIME_FALLBACK_TRAINING_PROFILE.example_generation_backend_name
    )
    assert config.evidence_backend_name is None
    assert config.scorer_backend_name is None
    assert config.score_policy_name is None
    assert (
        config.privacy_guard_name
        == RUNTIME_FALLBACK_TRAINING_PROFILE.privacy_guard_name
    )
    assert config.pseudo_label_algorithm_name is None
    assert config.acceptance_policy_name is None
    assert config.extras == {
        "query_ssl.strong_view_policy": FIXMATCH_QUERY_SSL_STRONG_VIEW_POLICY,
        "query_ssl.unlabeled_batch_size": 8,
        **{f"query_ssl.{key}": value for key, value in query_ssl_defaults.items()},
        "peft_classifier.delta_format": PEFT_ENCODER_DELTA_FORMAT_INLINE,
    }
    assert config.example_generation_backend_name == WEAK_STRONG_PAIR_EXAMPLE_BACKEND


def test_runtime_fallback_objective_builder_accepts_overrides() -> None:
    config = build_runtime_fallback_training_objective_config(
        overrides={
            "selection.confidence_threshold": 0.75,
            "score_top_k": 3,
            "training_backend.max_abs_delta": 0.02,
        }
    )

    assert config.score_top_k == 3
    assert config.extras["selection.confidence_threshold"] == 0.75
    assert config.extras["training_backend.max_abs_delta"] == 0.02


def test_runtime_fallback_selection_policy_builder_uses_profile() -> None:
    policy = build_runtime_fallback_training_selection_policy()

    assert policy.max_examples == RUNTIME_FALLBACK_TRAINING_PROFILE.max_examples


def test_runtime_fallback_secure_aggregation_builder_starts_disabled() -> None:
    config = build_runtime_fallback_secure_aggregation_config()

    assert config.required is False


def test_runtime_fallback_profile_exposes_round_task_runtime_defaults() -> None:
    assert RUNTIME_FALLBACK_TRAINING_PROFILE.local_epochs == 1
    assert RUNTIME_FALLBACK_TRAINING_PROFILE.batch_size == 8
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
                {"example_generation_backend_name": "peft_classifier_raw_rows"},
            )()
        )
        == "peft_classifier_raw_rows"
    )
