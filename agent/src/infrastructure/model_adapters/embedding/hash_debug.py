"""공통 hash debug 임베딩 어댑터 래퍼."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracemind_embedding.adapters.hash_debug import HashDebugEmbeddingAdapter

__all__ = ["HashDebugEmbeddingAdapter"]
