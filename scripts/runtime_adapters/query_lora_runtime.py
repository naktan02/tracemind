"""central query LoRA/SSL runner가 쓰는 agent-local trainer bridge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def set_query_lora_seed(seed: int) -> None:
    from agent.src.services.training.query_classifier_adaptation.training import (
        set_seed,
    )

    set_seed(seed)


def build_query_lora_label_index(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, int]]:
    from agent.src.services.training.query_classifier_adaptation.data import (
        build_label_index,
    )

    return build_label_index(rows)


def build_query_lora_dataloader(**kwargs: Any) -> Any:
    from agent.src.services.training.query_classifier_adaptation.data import (
        build_dataloader,
    )

    return build_dataloader(**kwargs)


def build_query_lora_multiview_dataloader(**kwargs: Any) -> Any:
    from agent.src.services.training.query_classifier_adaptation.data import (
        build_multiview_dataloader,
    )

    return build_multiview_dataloader(**kwargs)


def build_query_lora_model(**kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
    from agent.src.services.training.query_classifier_adaptation.modeling import (
        build_model,
    )

    return build_model(**kwargs)


def evaluate_query_lora_classifier(**kwargs: Any) -> dict[str, Any]:
    from agent.src.services.training.query_classifier_adaptation.training import (
        evaluate_classifier,
    )

    return evaluate_classifier(**kwargs)


def train_query_lora_classifier(
    **kwargs: Any,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    from agent.src.services.training.query_classifier_adaptation.training import (
        train_classifier,
    )

    return train_classifier(**kwargs)


def train_query_ssl_lora_classifier(
    **kwargs: Any,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    from agent.src.services.training.query_classifier_adaptation.training import (
        train_query_ssl_classifier,
    )

    return train_query_ssl_classifier(**kwargs)
