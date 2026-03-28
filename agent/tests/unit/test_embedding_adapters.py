"""Embedding adapter unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracemind_embedding.adapters.hash_debug import HashDebugEmbeddingAdapter
from tracemind_embedding.adapters.mxbai import MxbaiEmbeddingAdapter


def test_hash_debug_adapter_is_deterministic_and_normalized() -> None:
    adapter = HashDebugEmbeddingAdapter(vector_dim=32, normalize_embeddings=True)

    first = adapter.embed_texts(["panic attack at night"])[0]
    second = adapter.embed_texts(["panic attack at night"])[0]

    assert first == pytest.approx(second)
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_mxbai_adapter_uses_sentence_transformer_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs
            captured["device"] = None

        def to(self, device: str) -> None:
            captured["device"] = device

        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            captured["texts"] = texts
            captured["encode_kwargs"] = kwargs
            return [[0.1, 0.2], [0.3, 0.4]]

    adapter = MxbaiEmbeddingAdapter(
        model_id="mixedbread-ai/mxbai-embed-large-v1",
        revision="main",
        device="auto",
        batch_size=8,
        cache_dir="hf_cache",
        task_prefix="query: ",
        local_files_only=True,
    )
    monkeypatch.setattr(
        "tracemind_embedding.adapters.mxbai.resolve_runtime_device",
        lambda device: "cuda",
    )
    monkeypatch.setattr(
        MxbaiEmbeddingAdapter,
        "_load_sentence_transformer_class",
        lambda self: FakeSentenceTransformer,
    )

    embeddings = adapter.embed_texts(["hello", "world"])

    assert captured["model_id"] == "mixedbread-ai/mxbai-embed-large-v1"
    assert captured["kwargs"] == {
        "revision": "main",
        "cache_folder": "hf_cache",
        "local_files_only": True,
    }
    assert captured["device"] == "cuda"
    assert captured["texts"] == ["query: hello", "query: world"]
    assert captured["encode_kwargs"] == {
        "batch_size": 8,
        "normalize_embeddings": True,
        "convert_to_numpy": False,
        "show_progress_bar": False,
    }
    assert embeddings[0] == pytest.approx([0.1, 0.2])
    assert embeddings[1] == pytest.approx([0.3, 0.4])


def test_mxbai_adapter_raises_when_no_runtime_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MxbaiEmbeddingAdapter(model_id="mixedbread-ai/mxbai-embed-large-v1")
    monkeypatch.setattr(
        MxbaiEmbeddingAdapter,
        "_load_sentence_transformer_class",
        lambda self: None,
    )
    monkeypatch.setattr(
        MxbaiEmbeddingAdapter,
        "_get_transformers_runtime",
        lambda self: (_ for _ in ()).throw(
            RuntimeError("mxbai_large backend requires sentence_transformers or torch+transformers.")
        ),
    )

    with pytest.raises(RuntimeError, match="requires sentence_transformers or torch\\+transformers"):
        adapter.embed_texts(["hello"])
