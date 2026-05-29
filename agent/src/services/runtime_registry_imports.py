"""Runtime registry import helpers."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable

SKIPPED_REGISTRY_MODULE_PARTS = frozenset(
    {
        "base",
        "builtin_loader",
        "compatibility",
        "helpers",
        "models",
        "registry",
        "resolver",
    }
)


def import_runtime_module_for_name(
    *,
    package_name: str,
    registered_name: str,
) -> None:
    """등록 이름과 module naming convention으로 필요한 runtime module을 import한다."""

    normalized_name = registered_name.strip().lower().replace("-", "_")
    for module_name in _candidate_module_names(normalized_name):
        if _try_import_module(f"{package_name}.{module_name}"):
            return


def import_runtime_package_modules(*, package_name: str) -> None:
    """package-local runtime modules를 import해서 local decorators를 실행한다."""

    package = importlib.import_module(package_name)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return

    for module_info in pkgutil.walk_packages(package_paths, prefix=f"{package_name}."):
        relative_parts = module_info.name.removeprefix(f"{package_name}.").split(".")
        if any(part in SKIPPED_REGISTRY_MODULE_PARTS for part in relative_parts):
            continue
        importlib.import_module(module_info.name)


def _candidate_module_names(normalized_name: str) -> Iterable[str]:
    parts = normalized_name.split("_")
    for end_index in range(len(parts), 0, -1):
        yield "_".join(parts[:end_index])


def _try_import_module(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            return False
        raise
    return True
