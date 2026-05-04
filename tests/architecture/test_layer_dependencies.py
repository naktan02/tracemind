"""레이어 의존 규칙 아키텍처 테스트."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_SRC = REPO_ROOT / "shared" / "src"
METHODS_SRC = REPO_ROOT / "methods"
AGENT_SRC = REPO_ROOT / "agent" / "src"
MAIN_SERVER_SRC = REPO_ROOT / "main_server" / "src"
SCRIPTS_SRC = REPO_ROOT / "scripts"
SCRIPTS_RUNTIME_ADAPTER_SRC = SCRIPTS_SRC / "runtime_adapters"
LEGACY_SHARED_PROTOTYPE_BUILDER_PATHS = (
    SHARED_SRC / "services" / "prototypes" / "build_strategies.py",
    SHARED_SRC / "services" / "prototypes" / "prototype_pack_builder.py",
)
PROTOTYPE_BUILDING_SRC = REPO_ROOT / "methods" / "prototype" / "building"

TEMPORARY_MAIN_SERVER_AGENT_IMPORT_EXCEPTIONS: set[Path] = set()


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.py") if "__pycache__" not in path.parts
    )


def _collect_absolute_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module)
    return imports


def _relative_repo_path(path: Path) -> Path:
    return path.relative_to(REPO_ROOT)


def _find_forbidden_imports(
    *,
    root: Path,
    forbidden_prefixes: tuple[str, ...],
    ignored_roots: tuple[Path, ...] = (),
) -> list[tuple[Path, str]]:
    violations: list[tuple[Path, str]] = []
    for path in _iter_python_files(root):
        if any(path.is_relative_to(ignored_root) for ignored_root in ignored_roots):
            continue
        imports = _collect_absolute_imports(path)
        for imported_module in sorted(imports):
            if imported_module.startswith(forbidden_prefixes):
                violations.append((_relative_repo_path(path), imported_module))
    return violations


def _format_violations(violations: list[tuple[Path, str]]) -> str:
    return "\n".join(
        f"- {path}: {imported_module}" for path, imported_module in violations
    )


def test_shared_layer_does_not_import_runtime_layers() -> None:
    violations = _find_forbidden_imports(
        root=SHARED_SRC,
        forbidden_prefixes=(
            "agent.src",
            "main_server.src",
            "methods",
            "research",
            "scripts",
        ),
    )
    assert not violations, _format_violations(violations)


def test_prototype_builder_core_stays_in_methods_layer() -> None:
    existing_paths = [
        _relative_repo_path(path)
        for path in LEGACY_SHARED_PROTOTYPE_BUILDER_PATHS
        if path.exists()
    ]
    assert not existing_paths, (
        "prototype builder 알고리즘 core는 methods/prototype/building에 둔다. "
        f"legacy shared paths={sorted(str(path) for path in existing_paths)}"
    )


def test_prototype_building_keeps_strategy_files_separate() -> None:
    monolith_path = PROTOTYPE_BUILDING_SRC / "build_strategies.py"
    assert not monolith_path.exists(), (
        "prototype builder strategy는 base/single/kmeans/dbscan 파일로 나눈다. "
        f"monolith path={_relative_repo_path(monolith_path)}"
    )


def test_methods_layer_does_not_import_runtime_or_research_layers() -> None:
    violations = _find_forbidden_imports(
        root=METHODS_SRC,
        forbidden_prefixes=("agent.src", "main_server.src", "research", "scripts"),
    )
    assert not violations, _format_violations(violations)


def test_agent_layer_does_not_import_main_server_or_scripts() -> None:
    violations = _find_forbidden_imports(
        root=AGENT_SRC,
        forbidden_prefixes=("main_server.src", "research", "scripts"),
    )
    assert not violations, _format_violations(violations)


def test_main_server_layer_does_not_import_scripts() -> None:
    violations = _find_forbidden_imports(
        root=MAIN_SERVER_SRC,
        forbidden_prefixes=("research", "scripts"),
    )
    assert not violations, _format_violations(violations)


def test_main_server_agent_imports_are_limited_to_documented_exceptions() -> None:
    violations = _find_forbidden_imports(
        root=MAIN_SERVER_SRC,
        forbidden_prefixes=("agent.src",),
    )
    actual_exception_paths = {path for path, _ in violations}
    assert actual_exception_paths == TEMPORARY_MAIN_SERVER_AGENT_IMPORT_EXCEPTIONS, (
        "main_server -> agent 직접 의존은 현재 없어야 한다.\n"
        f"actual={sorted(str(path) for path in actual_exception_paths)}\n"
        "expected="
        f"{sorted(str(path) for path in TEMPORARY_MAIN_SERVER_AGENT_IMPORT_EXCEPTIONS)}"
    )


def test_scripts_runtime_imports_stay_behind_documented_bridges() -> None:
    violations = _find_forbidden_imports(
        root=SCRIPTS_SRC,
        forbidden_prefixes=("agent.src", "main_server.src"),
        ignored_roots=(SCRIPTS_RUNTIME_ADAPTER_SRC,),
    )
    assert not violations, (
        "scripts는 agent/main_server 내부를 직접 import하지 않는다. "
        "runtime bridge는 scripts/runtime_adapters에 둔다.\n"
        f"{_format_violations(violations)}"
    )
