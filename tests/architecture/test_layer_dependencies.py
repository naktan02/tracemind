"""레이어 의존 규칙 아키텍처 테스트."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_SRC = REPO_ROOT / "shared" / "src"
AGENT_SRC = REPO_ROOT / "agent" / "src"
MAIN_SERVER_SRC = REPO_ROOT / "main_server" / "src"

TEMPORARY_MAIN_SERVER_AGENT_IMPORT_EXCEPTIONS: set[Path] = set()


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


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
) -> list[tuple[Path, str]]:
    violations: list[tuple[Path, str]] = []
    for path in _iter_python_files(root):
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
        forbidden_prefixes=("agent.src", "main_server.src", "scripts"),
    )
    assert not violations, _format_violations(violations)


def test_agent_layer_does_not_import_main_server_or_scripts() -> None:
    violations = _find_forbidden_imports(
        root=AGENT_SRC,
        forbidden_prefixes=("main_server.src", "scripts"),
    )
    assert not violations, _format_violations(violations)


def test_main_server_layer_does_not_import_scripts() -> None:
    violations = _find_forbidden_imports(
        root=MAIN_SERVER_SRC,
        forbidden_prefixes=("scripts",),
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
        f"expected={sorted(str(path) for path in TEMPORARY_MAIN_SERVER_AGENT_IMPORT_EXCEPTIONS)}"
    )
