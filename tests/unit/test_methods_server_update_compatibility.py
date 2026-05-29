"""Server update/state compatibility preflight tests."""

from __future__ import annotations

import pytest

from methods.adaptation.server_update_compatibility import (
    require_server_compatible_update_payload,
)
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_peft_classifier_delta_payload,
    make_peft_classifier_state_payload,
)

_BACKBONE = {
    "backbone_model_id": "mixedbread-ai/mxbai-embed-large-v1",
    "backbone_revision": "main",
    "tokenizer_model_id": "mixedbread-ai/mxbai-embed-large-v1",
    "tokenizer_revision": "main",
    "pooling": "mean",
    "max_length": 256,
    "task_prefix": "",
}
_LORA_CONFIG = {
    "peft_adapter_name": "lora",
    "rank": 8,
    "alpha": 16,
    "dropout": 0.1,
    "bias": "none",
    "target_modules": "all-linear",
    "use_rslora": False,
}
_PEFT_CONFIG = {
    "peft_adapter_name": "lora",
    "parameters": _LORA_CONFIG,
}


def test_unknown_adapter_kind_compatibility_defaults_to_noop() -> None:
    state = SharedAdapterStatePayload(
        schema_version="test_state.v1",
        adapter_kind="test_family",
        model_id="tracemind-embed",
        model_revision="rev_000",
        training_scope="adapter_only",
        updated_at="2026-04-01T00:00:00Z",
    )
    update = SharedAdapterUpdatePayload(
        schema_version="test_update.v1",
        adapter_kind="test_family",
        model_id="tracemind-embed",
        base_model_revision="rev_000",
        training_scope="adapter_only",
        example_count=2,
    )

    require_server_compatible_update_payload(
        update_payload=update,
        active_state=state,
    )


def test_peft_classifier_compatibility_accepts_matching_active_state() -> None:
    state = _peft_state()
    update = _peft_update()

    require_server_compatible_update_payload(
        update_payload=update,
        active_state=state,
    )


@pytest.mark.parametrize(
    ("field_name", "update_kwargs"),
    [
        ("model_id", {"model_id": "other-model"}),
        ("base_model_revision", {"base_model_revision": "rev_999"}),
        ("training_scope", {"training_scope": "head_only"}),
        ("backbone", {"backbone": {**_BACKBONE, "backbone_revision": "v2"}}),
        (
            "peft_adapter_config",
            {
                "peft_adapter_config": {
                    **_PEFT_CONFIG,
                    "parameters": {**_LORA_CONFIG, "rank": 4},
                }
            },
        ),
        (
            "label_schema",
            {
                "label_schema": ["anxiety", "depression"],
                "classifier_head_weight_deltas": {
                    "anxiety": [0.1, 0.0],
                    "depression": [0.0, -0.1],
                },
                "classifier_head_bias_deltas": {
                    "anxiety": 0.01,
                    "depression": -0.01,
                },
            },
        ),
    ],
)
def test_peft_classifier_compatibility_rejects_payload_drift(
    field_name: str,
    update_kwargs: dict[str, object],
) -> None:
    state = _peft_state()
    update = _peft_update(**update_kwargs)

    with pytest.raises(ValueError, match=field_name):
        require_server_compatible_update_payload(
            update_payload=update,
            active_state=state,
        )


def _peft_state():
    return make_peft_classifier_state_payload(
        model_id="mxbai-peft-classifier",
        model_revision="rev_000",
        training_scope="adapter_only",
        backbone=_BACKBONE,
        peft_adapter_config=_PEFT_CONFIG,
        label_schema=["anxiety", "normal"],
    )


def _peft_update(**overrides: object):
    values = {
        "model_id": "mxbai-peft-classifier",
        "base_model_revision": "rev_000",
        "training_scope": "adapter_only",
        "backbone": _BACKBONE,
        "peft_adapter_config": _PEFT_CONFIG,
        "label_schema": ["anxiety", "normal"],
        "example_count": 2,
        "peft_parameter_deltas": {"encoder.q_proj.lora_A": [0.1, -0.2]},
        "classifier_head_weight_deltas": {
            "anxiety": [0.1, 0.0],
            "normal": [0.0, -0.1],
        },
        "classifier_head_bias_deltas": {"anxiety": 0.01, "normal": -0.01},
        "delta_format": "inline_delta",
    }
    values.update(overrides)
    return make_peft_classifier_delta_payload(**values)
