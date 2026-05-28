"""Config-declared runtime callable loading helpers."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import Any


@cache
def load_configured_callable(callable_path: str, *, field_name: str) -> Any:
    """Hydra config가 선언한 fully-qualified callable을 로드한다."""

    module_name, separator, function_name = callable_path.rpartition(".")
    if not separator or not module_name or not function_name:
        raise ValueError(
            f"{field_name} must be a fully qualified function path: "
            f"{callable_path!r}."
        )
    module = import_module(module_name)
    loaded_callable = getattr(module, function_name, None)
    if not callable(loaded_callable):
        raise ValueError(
            f"{field_name} must point to a callable: {callable_path!r}."
        )
    return loaded_callable
