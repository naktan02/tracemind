"""Prototype strategy 실험용 샘플링 유틸리티."""

from __future__ import annotations

import numpy as np


def sample_index_array(
    count: int,
    *,
    limit: int | None,
    seed: int,
) -> np.ndarray:
    """count 개수 중 limit 개 인덱스를 재현 가능하게 샘플링한다."""
    indices = np.arange(count)
    if limit is None or count <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))
