"""스크립트 임베딩 profile 공통 설정 tests."""

from __future__ import annotations

from argparse import Namespace

from scripts.common.embedding_profiles import (
    infer_embedding_profile,
    resolve_embedding_settings,
    resolve_embedding_settings_from_args,
)
from scripts.experiments.prototype_strategy.config import load_experiment_config
from scripts.experiments.prototype_strategy.sweep_config import (
    load_threshold_sweep_config,
)


def test_resolve_embedding_settings_uses_profile_defaults() -> None:
    settings = resolve_embedding_settings(profile="mxbai")

    assert settings.profile == "mxbai"
    assert settings.backend == "transformers_mxbai"
    assert settings.model_id == "mixedbread-ai/mxbai-embed-large-v1"
    assert settings.revision == "main"


def test_resolve_embedding_settings_from_args_allows_model_override() -> None:
    args = Namespace(
        embedding_profile="mxbai",
        backend=None,
        embedding_model_id="intfloat/e5-large-v2",
        embedding_model_revision="refs/pr/1",
        batch_size=None,
        cache_dir=None,
        device="cuda",
        task_prefix=None,
        local_files_only=True,
        hash_dim=None,
    )

    settings = resolve_embedding_settings_from_args(args)

    assert settings.profile == "mxbai"
    assert settings.backend == "transformers_mxbai"
    assert settings.model_id == "intfloat/e5-large-v2"
    assert settings.revision == "refs/pr/1"
    assert settings.device == "cuda"
    assert settings.local_files_only is True


def test_infer_embedding_profile_uses_known_model_id() -> None:
    assert (
        infer_embedding_profile(model_id="hash_debug", fallback="mxbai")
        == "hash_debug"
    )


def test_prototype_strategy_default_config_resolves_profile() -> None:
    config = load_experiment_config()

    assert config.embedding.profile == "mxbai"
    assert config.embedding.backend == "transformers_mxbai"
    assert config.embedding.model_id == "mixedbread-ai/mxbai-embed-large-v1"


def test_threshold_sweep_config_allows_profile_override() -> None:
    config = load_threshold_sweep_config(
        overrides=(
            "embedding.profile=hash_debug",
            "embedding.hash_dim=64",
        )
    )

    assert config.embedding.profile == "hash_debug"
    assert config.embedding.backend == "hash_debug"
    assert config.embedding.model_id == "hash_debug"
    assert config.embedding.hash_dim == 64
