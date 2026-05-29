"""API router 파일과 app 조립 경로 parity guard."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_agent_api_router_modules_are_registered() -> None:
    _assert_router_modules_registered(
        api_root=REPO_ROOT / "agent" / "src" / "api",
        main_path=REPO_ROOT / "agent" / "src" / "api" / "main.py",
        package_prefix="agent.src.api",
    )


def test_main_server_api_router_modules_are_registered() -> None:
    _assert_router_modules_registered(
        api_root=REPO_ROOT / "main_server" / "src" / "api",
        main_path=REPO_ROOT / "main_server" / "src" / "api" / "main.py",
        package_prefix="main_server.src.api",
    )


def _assert_router_modules_registered(
    *,
    api_root: Path,
    main_path: Path,
    package_prefix: str,
) -> None:
    router_modules = _router_module_names(api_root)
    registered_modules = _registered_router_modules(
        main_path=main_path,
        package_prefix=package_prefix,
    )
    missing = sorted(router_modules - registered_modules)
    stale_imports = sorted(registered_modules - router_modules)

    assert not missing, (
        "api/*.py에 router를 둔 Module은 app main에서 include_router까지 닫는다. "
        "행동 없는 placeholder router는 만들지 않는다.\n"
        f"missing={missing}"
    )
    assert not stale_imports, (
        "app main은 존재하지 않는 router Module을 import하지 않는다.\n"
        f"stale_imports={stale_imports}"
    )


def _router_module_names(api_root: Path) -> set[str]:
    modules: set[str] = set()
    for path in sorted(api_root.glob("*.py")):
        if path.name in {"__init__.py", "main.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _defines_router(tree):
            modules.add(path.stem)
    return modules


def _registered_router_modules(
    *,
    main_path: Path,
    package_prefix: str,
) -> set[str]:
    tree = ast.parse(main_path.read_text(encoding="utf-8"), filename=str(main_path))
    router_alias_to_module: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None or not node.module.startswith(f"{package_prefix}."):
            continue
        module_name = node.module.removeprefix(f"{package_prefix}.").split(".", 1)[0]
        for alias in node.names:
            if alias.name == "router":
                router_alias_to_module[alias.asname or alias.name] = module_name

    included_aliases = {
        call.args[0].id
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "include_router"
        and call.args
        and isinstance(call.args[0], ast.Name)
    }
    return {
        module_name
        for alias_name, module_name in router_alias_to_module.items()
        if alias_name in included_aliases
    }


def _defines_router(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "router"
            for target in node.targets
        ):
            return True
    return False
