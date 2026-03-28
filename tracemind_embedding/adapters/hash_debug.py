"""디버그용 해시 임베딩 어댑터."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class HashDebugEmbeddingAdapter:
    """외부 모델 없이 deterministic pseudo-embedding을 생성한다."""

    vector_dim: int = 256
    normalize_embeddings: bool = True

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.vector_dim
            tokens = text.split() or [text]
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], byteorder="big") % self.vector_dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                magnitude = 1.0 + (digest[5] / 255.0)
                vector[index] += sign * magnitude

            if self.normalize_embeddings:
                norm = math.sqrt(sum(value * value for value in vector))
                if norm > 0.0:
                    vector = [value / norm for value in vector]
            embeddings.append(vector)

        return embeddings
