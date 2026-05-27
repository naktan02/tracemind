"""Query text tokenization cache와 padding helper."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from methods.common.runtime_resources import RuntimeResourceCache

TEXT_TOKENIZATION_CACHE_RESOURCE_KEY = "query_text_views:text_tokenization_cache:v1"


@dataclass(frozen=True, slots=True)
class TextTokenizationCacheKey:
    """tokenizer config와 text 기준 tokenization cache key."""

    namespace: str
    max_length: int
    text: str


@dataclass(slots=True)
class TextTokenizationCache:
    """선택된 text의 tokenizer 결과를 run-local로 재사용한다."""

    _entries: dict[
        TextTokenizationCacheKey,
        tuple[tuple[int, ...], tuple[int, ...]],
    ] = field(default_factory=dict)

    def encode(
        self,
        *,
        text: str,
        tokenizer: Any,
        namespace: str,
        max_length: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        key = TextTokenizationCacheKey(
            namespace=str(namespace),
            max_length=int(max_length),
            text=str(text),
        )
        cached = self._entries.get(key)
        if cached is not None:
            return cached
        encoded = tokenizer(
            str(text),
            padding=False,
            truncation=True,
            max_length=int(max_length),
            return_attention_mask=True,
        )
        input_ids = tuple(int(value) for value in encoded["input_ids"])
        attention_mask = tuple(int(value) for value in encoded["attention_mask"])
        self._entries[key] = (input_ids, attention_mask)
        return input_ids, attention_mask


def resolve_text_tokenization_cache(
    runtime_resource_cache: RuntimeResourceCache | None,
) -> TextTokenizationCache | None:
    """runtime cache에서 text tokenization cache를 가져오거나 생성한다."""

    if runtime_resource_cache is None or not hasattr(
        runtime_resource_cache, "get_resource"
    ):
        return None
    cached = runtime_resource_cache.get_resource(TEXT_TOKENIZATION_CACHE_RESOURCE_KEY)
    if cached is None:
        tokenization_cache = TextTokenizationCache()
        runtime_resource_cache.set_resource(
            TEXT_TOKENIZATION_CACHE_RESOURCE_KEY,
            tokenization_cache,
        )
        return tokenization_cache
    if not isinstance(cached, TextTokenizationCache):
        raise TypeError(
            "Runtime resource cache key is occupied by non-tokenization cache: "
            f"{type(cached)!r}."
        )
    return cached


def encode_texts(
    texts: Sequence[str],
    *,
    tokenizer: Any,
    max_length: int,
    tokenization_cache: TextTokenizationCache | None,
    tokenization_cache_namespace: str | None,
) -> dict[str, torch.Tensor]:
    """batch tokenizer path와 run-local cache path를 같은 tensor shape로 정규화한다."""

    if tokenization_cache is None:
        return tokenizer(
            [str(text) for text in texts],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    namespace = (
        str(tokenization_cache_namespace)
        if tokenization_cache_namespace is not None
        else _tokenizer_namespace(tokenizer)
    )
    encoded = [
        tokenization_cache.encode(
            text=str(text),
            tokenizer=tokenizer,
            namespace=namespace,
            max_length=max_length,
        )
        for text in texts
    ]
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    padding_side = str(getattr(tokenizer, "padding_side", "right") or "right")
    return _pad_encoded_texts(
        encoded=encoded,
        pad_token_id=pad_token_id,
        padding_side=padding_side,
    )


def _pad_encoded_texts(
    *,
    encoded: Sequence[tuple[Sequence[int], Sequence[int]]],
    pad_token_id: int,
    padding_side: str,
) -> dict[str, torch.Tensor]:
    max_size = max((len(input_ids) for input_ids, _mask in encoded), default=0)
    input_ids_tensor = torch.full(
        (len(encoded), max_size),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask_tensor = torch.zeros(
        (len(encoded), max_size),
        dtype=torch.long,
    )
    for row_index, (input_ids, attention_mask) in enumerate(encoded):
        size = len(input_ids)
        if padding_side == "left":
            start = max_size - size
            stop = max_size
        else:
            start = 0
            stop = size
        input_ids_tensor[row_index, start:stop] = torch.tensor(
            list(input_ids),
            dtype=torch.long,
        )
        attention_mask_tensor[row_index, start:stop] = torch.tensor(
            list(attention_mask),
            dtype=torch.long,
        )
    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
    }


def _tokenizer_namespace(tokenizer: Any) -> str:
    return str(getattr(tokenizer, "name_or_path", type(tokenizer).__qualname__))
