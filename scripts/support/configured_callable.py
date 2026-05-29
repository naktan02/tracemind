"""Config가 선언한 callable을 로드하는 작은 helper."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def load_configured_callable(callable_path: str, *, field_name: str) -> Any:
    """fully qualified callable path를 import한다."""

    module_name, separator, attribute_name = callable_path.strip().rpartition(".")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            f"{field_name} must be a fully qualified callable path: "
            f"{callable_path!r}."
        )
    target = getattr(import_module(module_name), attribute_name)
    if not callable(target):
        raise TypeError(
            f"{field_name} must point to a callable: {callable_path!r}."
        )
    return target
