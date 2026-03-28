"""Embedding adapter factory unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracemind_embedding.adapters.hash_debug import HashDebugEmbeddingAdapter
from tracemind_embedding.adapters.mxbai import MxbaiEmbeddingAdapter
from tracemind_embedding.base import EmbeddingAdapterSpec
from tracemind_embedding.factory import EmbeddingAdapterFactory


def test_factory_creates_hash_debug_adapter() -> None:
    adapter = EmbeddingAdapterFactory.create(
        EmbeddingAdapterSpec(
            backend="hash_debug",
            model_id="hash_debug/mxbai_shape_proxy",
            hash_dim=64,
        )
    )

    assert isinstance(adapter, HashDebugEmbeddingAdapter)
    assert adapter.vector_dim == 64


def test_factory_creates_mxbai_adapter_from_legacy_alias() -> None:
    adapter = EmbeddingAdapterFactory.create(
        EmbeddingAdapterSpec(
            backend="transformers_mxbai",
            model_id="mixedbread-ai/mxbai-embed-large-v1",
            revision="main",
            batch_size=4,
        )
    )

    assert isinstance(adapter, MxbaiEmbeddingAdapter)
    assert adapter.model_id == "mixedbread-ai/mxbai-embed-large-v1"
    assert adapter.batch_size == 4
    assert adapter.device == "auto"


def test_factory_rejects_unsupported_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding backend"):
        EmbeddingAdapterFactory.create(
            EmbeddingAdapterSpec(
                backend="unknown_backend",
                model_id="unused",
            )
        )
