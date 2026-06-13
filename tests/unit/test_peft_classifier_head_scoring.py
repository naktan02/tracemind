"""PEFT classifier-head inference scoring tests."""

from __future__ import annotations

import pytest

from methods.adaptation.peft_text_encoder.scoring import (
    PEFT_CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    build_peft_classifier_head_scoring_assets,
)
from methods.adaptation.scoring_registry import build_shared_adapter_scoring_backend
from shared.src.contracts.adapter_contract_families.factories import (
    make_peft_classifier_state_payload,
)
from shared.src.contracts.scoring_contracts import ScoringConfigPayload


def test_peft_classifier_head_logits_scores_materialized_head_artifact() -> None:
    state = _state()
    assets = build_peft_classifier_head_scoring_assets(
        classifier_head_artifact={
            "classifier_head_weights": {
                "anxiety": [0.2, -0.1],
                "normal": [-0.4, 0.3],
            },
            "classifier_head_biases": {
                "anxiety": 0.05,
                "normal": -0.02,
            },
        },
        label_schema=state.labels,
    )
    backend = build_shared_adapter_scoring_backend(
        PEFT_CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
        scoring_config=ScoringConfigPayload(
            scorer_backend_name=PEFT_CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
        ),
    )

    scores = backend.score([0.5, 0.25], assets, shared_state=state)

    assert scores == pytest.approx(
        {
            "anxiety": 0.125,
            "normal": -0.145,
        }
    )


def test_peft_classifier_head_logits_requires_head_assets() -> None:
    backend = build_shared_adapter_scoring_backend(
        PEFT_CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
        scoring_config=ScoringConfigPayload(
            scorer_backend_name=PEFT_CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
        ),
    )

    with pytest.raises(ValueError, match="classifier_head_weights"):
        backend.score([1.0, 0.0], {}, shared_state=_state())


def _state():
    return make_peft_classifier_state_payload(
        model_id="tracemind-main",
        model_revision="rev_001",
        backbone={
            "backbone_model_id": "mixedbread-ai/mxbai-embed-large-v1",
            "backbone_revision": "main",
            "tokenizer_model_id": "mixedbread-ai/mxbai-embed-large-v1",
            "tokenizer_revision": "main",
            "max_length": 256,
        },
        peft_adapter_config={
            "peft_adapter_name": "lora",
            "parameters": {"rank": 8},
        },
        label_schema=("anxiety", "normal"),
        classifier_head_artifact_ref=(
            "server-aggregate://peft_classifier/rev_001/classifier_head"
        ),
    )
