"""Query SSL algorithm-local state pack/unpack helper."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, TypeAlias

import torch
from torch import Tensor

QUERY_SSL_ALGORITHM_STATE_SCHEMA_VERSION = "query_ssl_algorithm_state.v1"
JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


def build_stateless_query_ssl_algorithm_state(
    *,
    algorithm_name: str,
) -> dict[str, Any]:
    """state가 없는 algorithm의 resume payload를 만든다."""

    return {
        "schema_version": QUERY_SSL_ALGORITHM_STATE_SCHEMA_VERSION,
        "algorithm_name": algorithm_name,
        "stateful": False,
    }


def build_query_ssl_algorithm_state(
    *,
    algorithm_name: str,
    configured: bool,
    metadata: Mapping[str, Any] | None = None,
    tensors: Mapping[str, Tensor | None] | None = None,
) -> dict[str, Any]:
    """algorithm-local state를 checkpoint-safe payload로 정규화한다."""

    state: dict[str, Any] = {
        "schema_version": QUERY_SSL_ALGORITHM_STATE_SCHEMA_VERSION,
        "algorithm_name": algorithm_name,
        "stateful": True,
        "configured": bool(configured),
    }
    if metadata is not None:
        state.update(dict(metadata))
    if tensors is not None:
        for name, tensor in tensors.items():
            state[name] = None if tensor is None else tensor.detach().cpu().clone()
    return state


def require_query_ssl_algorithm_state(
    *,
    state: Mapping[str, Any],
    algorithm_name: str,
) -> Mapping[str, Any]:
    """저장된 state가 현재 algorithm과 같은 schema인지 검증한다."""

    schema_version = state.get("schema_version")
    if schema_version is not None and str(schema_version) != (
        QUERY_SSL_ALGORITHM_STATE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported Query SSL algorithm state schema_version: {schema_version!r}."
        )
    if str(state.get("algorithm_name", algorithm_name)) != algorithm_name:
        raise ValueError(
            f"{algorithm_name} cannot load state for a different algorithm."
        )
    return state


def is_configured_query_ssl_algorithm_state(state: Mapping[str, Any]) -> bool:
    """configured=false state는 configure 전 checkpoint였음을 뜻한다."""

    return bool(state.get("configured", True))


def require_matching_int_state_value(
    *,
    state: Mapping[str, Any],
    field_name: str,
    expected: int,
    algorithm_name: str,
) -> int:
    """dataset shape 같은 integer metadata가 현재 실행과 맞는지 확인한다."""

    actual = int(state.get(field_name, expected))
    if actual != int(expected):
        raise ValueError(
            f"{algorithm_name} state {field_name} does not match configured dataset."
        )
    return actual


def load_tensor_state_field(
    *,
    state: Mapping[str, Any],
    field_name: str,
    device: torch.device,
    algorithm_name: str,
    allow_none: bool = False,
) -> Tensor | None:
    """checkpoint tensor field를 현재 device로 복원한다."""

    value = state.get(field_name)
    if value is None and allow_none:
        return None
    if not isinstance(value, Tensor):
        raise ValueError(f"{algorithm_name} state requires tensor {field_name}.")
    return value.detach().clone().to(device)


def export_query_ssl_algorithm_state(algorithm: Any) -> Mapping[str, Any]:
    """stateful algorithm이면 resume용 state를 내보내고 아니면 빈 state를 반환한다."""

    export_state = getattr(algorithm, "export_state", None)
    if callable(export_state):
        return dict(export_state())
    return build_stateless_query_ssl_algorithm_state(
        algorithm_name=str(algorithm.algorithm_name),
    )


def export_query_ssl_algorithm_report_state_summary(
    algorithm: Any,
) -> Mapping[str, JsonValue]:
    """report/manifest에 넣을 JSON-safe algorithm runtime state 요약을 만든다."""

    export_report_state_summary = getattr(
        algorithm,
        "export_report_state_summary",
        None,
    )
    if callable(export_report_state_summary):
        safe_summary = _to_json_safe_value(export_report_state_summary())
    else:
        safe_summary = _to_json_safe_value(export_query_ssl_algorithm_state(algorithm))
    if isinstance(safe_summary, dict):
        return safe_summary
    return {"value": safe_summary}


def load_query_ssl_algorithm_state(
    algorithm: Any,
    state: Mapping[str, Any],
) -> None:
    """stateful algorithm이면 저장된 state를 복원한다."""

    load_state = getattr(algorithm, "load_state", None)
    if callable(load_state):
        load_state(state)


def _to_json_safe_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return _finite_float_or_none(value)
    if isinstance(value, Tensor):
        return _summarize_tensor_for_report(value)
    if isinstance(value, Mapping):
        return {str(key): _to_json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_json_safe_value(item) for item in value]
    return {"type": type(value).__name__}


def _summarize_tensor_for_report(tensor: Tensor) -> JsonValue:
    detached = tensor.detach().cpu()
    if detached.ndim == 0:
        scalar = detached.item()
        if isinstance(scalar, bool | int | str):
            return scalar
        return _finite_float_or_none(float(scalar))

    summary: dict[str, JsonValue] = {
        "type": "tensor",
        "shape": [int(dim) for dim in detached.shape],
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
    }
    if detached.numel() == 0:
        summary.update({"mean": None, "std": None, "min": None, "max": None})
        return summary

    numeric = detached.float()
    summary.update(
        {
            "mean": _finite_float_or_none(float(numeric.mean().item())),
            "std": _finite_float_or_none(float(numeric.std(unbiased=False).item())),
            "min": _finite_float_or_none(float(numeric.min().item())),
            "max": _finite_float_or_none(float(numeric.max().item())),
        }
    )
    return summary


def _finite_float_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None
