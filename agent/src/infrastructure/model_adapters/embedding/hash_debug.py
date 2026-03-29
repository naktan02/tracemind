"""디버그/테스트용 해시 기반 의사 임베딩 어댑터."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class HashDebugEmbeddingAdapter:
    """결정적 해시 기반 의사 임베딩 생성기.

    실제 모델 없이 파이프라인 smoke test가 가능하도록
    텍스트의 SHA-256 해시를 기반으로 고정 차원 벡터를 생성한다.
    """

    dim: int = 256

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._hash_embed(text) for text in texts]

    def _hash_embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # 해시 바이트를 반복 확장해 dim 차원 float 벡터 생성
        repeated = (digest * ((self.dim * 4 // len(digest)) + 1))[: self.dim * 4]
        values = list(struct.unpack(f"<{self.dim}f", repeated))

        # NaN/Inf 방지: 안전한 범위로 클리핑 후 L2 정규화
        import math

        safe = [max(-1e6, min(1e6, v)) if math.isfinite(v) else 0.0 for v in values]
        norm = math.sqrt(sum(v * v for v in safe)) or 1.0
        return [v / norm for v in safe]
