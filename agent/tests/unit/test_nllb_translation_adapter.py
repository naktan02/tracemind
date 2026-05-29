"""NLLB translation adapter tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from agent.src.infrastructure.model_adapters.translation.nllb import (
    NllbTranslationAdapter,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.src_lang = None
        self.lang_code_to_id = {"fra_Latn": 17}

    def __call__(self, texts, **_kwargs):
        batch = len(texts)
        return {
            "input_ids": torch.ones((batch, 2), dtype=torch.long),
            "attention_mask": torch.ones((batch, 2), dtype=torch.long),
        }

    def batch_decode(self, generated, **_kwargs):
        return [f"decoded::{index}" for index in range(generated.shape[0])]

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"fra_Latn": 17}[token]


class _FakeModel:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, object]] = []
        self.generation_config = SimpleNamespace(max_length=200)

    def to(self, _device: str):
        return self

    def eval(self) -> None:
        return None

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        batch_size = int(input_ids.shape[0])
        return torch.ones((batch_size, 3), dtype=torch.long)


class _FakeAutoTokenizer:
    call_count = 0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        cls.call_count += 1
        return _FakeTokenizer()


class _FakeAutoModelForSeq2SeqLM:
    call_count = 0
    last_kwargs = None

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        cls.call_count += 1
        cls.last_kwargs = dict(_kwargs)
        return _FakeModel()


def test_nllb_translation_adapter_translates_and_reuses_cached_bundle(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "agent.src.infrastructure.model_adapters.translation.nllb."
        "_require_transformer_seq2seq_stack",
        lambda: (_FakeAutoModelForSeq2SeqLM, _FakeAutoTokenizer),
    )
    NllbTranslationAdapter._MODEL_CACHE.clear()

    adapter = NllbTranslationAdapter(
        model_id="facebook/nllb-200-distilled-600M",
        revision="main",
        source_lang="eng_Latn",
        target_lang="fra_Latn",
        device="cpu",
        batch_size=2,
        local_files_only=True,
    )

    outputs = adapter.translate_texts(["hello", "world"])
    _, cached_model = adapter._load_bundle()

    assert outputs == ["decoded::0", "decoded::1"]
    generation_config = cached_model.generate_calls[0]["generation_config"]
    assert generation_config.forced_bos_token_id == 17
    assert generation_config.max_length is None
    assert _FakeAutoModelForSeq2SeqLM.last_kwargs["torch_dtype"] == torch.float32
    assert _FakeAutoTokenizer.call_count == 1
    assert _FakeAutoModelForSeq2SeqLM.call_count == 1

    second_adapter = NllbTranslationAdapter(
        model_id="facebook/nllb-200-distilled-600M",
        revision="main",
        source_lang="eng_Latn",
        target_lang="fra_Latn",
        device="cpu",
        batch_size=2,
        local_files_only=True,
    )

    second_outputs = second_adapter.translate_texts(["again"])

    assert second_outputs == ["decoded::0"]
    assert _FakeAutoTokenizer.call_count == 1
    assert _FakeAutoModelForSeq2SeqLM.call_count == 1


def test_nllb_translation_adapter_uses_float16_for_cuda_auto_dtype(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "agent.src.infrastructure.model_adapters.translation.nllb."
        "_require_transformer_seq2seq_stack",
        lambda: (_FakeAutoModelForSeq2SeqLM, _FakeAutoTokenizer),
    )
    NllbTranslationAdapter._MODEL_CACHE.clear()

    adapter = NllbTranslationAdapter(
        model_id="facebook/nllb-200-distilled-600M",
        revision="main",
        source_lang="eng_Latn",
        target_lang="fra_Latn",
        device="cuda",
        batch_size=2,
        local_files_only=True,
    )

    adapter._load_bundle()

    assert adapter.resolved_torch_dtype() == "float16"
    assert _FakeAutoModelForSeq2SeqLM.last_kwargs["torch_dtype"] == torch.float16
