"""Prototype score policy lookup/decorator registry."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable

from methods.prototype.scoring.base import (
    PrototypeScorePolicy,
    PrototypeScorePolicyFactory,
)

_SCORE_POLICY_PACKAGE = "methods.prototype.scoring.score_policies"
_PROTOTYPE_SCORE_POLICY_REGISTRY: dict[str, PrototypeScorePolicyFactory] = {}
_SCORE_POLICY_MODULES_LOADED = False


def register_prototype_score_policy(
    *policy_names: str,
) -> Callable[[PrototypeScorePolicyFactory], PrototypeScorePolicyFactory]:
    """이름으로 prototype score policy factory를 등록하는 decorator."""

    def _decorator(
        factory: PrototypeScorePolicyFactory,
    ) -> PrototypeScorePolicyFactory:
        names = tuple(policy_name.strip().lower() for policy_name in policy_names)
        if not names:
            raise ValueError("prototype score policy names must not be empty.")
        for policy_name in names:
            if not policy_name:
                raise ValueError("prototype score policy names must not be empty.")
            _PROTOTYPE_SCORE_POLICY_REGISTRY[policy_name] = factory
        return factory

    return _decorator


def build_prototype_score_policy(
    policy_name: str,
    *,
    top_k: int | None = None,
) -> PrototypeScorePolicy:
    """정책 이름으로 prototype score policy를 생성한다."""

    _import_score_policy_modules()
    normalized_name = policy_name.strip().lower()
    factory = _PROTOTYPE_SCORE_POLICY_REGISTRY.get(normalized_name)
    if factory is not None:
        return factory(top_k)
    raise ValueError(f"Unsupported prototype score policy: {policy_name}.")


def _import_score_policy_modules() -> None:
    global _SCORE_POLICY_MODULES_LOADED
    if _SCORE_POLICY_MODULES_LOADED:
        return

    package = importlib.import_module(_SCORE_POLICY_PACKAGE)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        _SCORE_POLICY_MODULES_LOADED = True
        return

    for module_info in pkgutil.iter_modules(package_paths):
        importlib.import_module(f"{_SCORE_POLICY_PACKAGE}.{module_info.name}")
    _SCORE_POLICY_MODULES_LOADED = True
