"""Fixed-feature 분류용 feature space 생성."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer


def build_feature_space(config: Mapping[str, Any]) -> Any:
    """Hydra leaf mapping에서 feature transformer를 생성한다."""

    name = str(config.get("name", ""))
    if name != "tfidf_word":
        raise ValueError(f"Unsupported fixed feature space: {name}")
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(
            int(config.get("ngram_min", 1)),
            int(config.get("ngram_max", 2)),
        ),
        min_df=int(config.get("min_df", 2)),
        max_df=float(config.get("max_df", 0.95)),
        sublinear_tf=bool(config.get("sublinear_tf", True)),
        lowercase=bool(config.get("lowercase", True)),
        strip_accents=str(config.get("strip_accents", "unicode")),
    )
