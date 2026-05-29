"""family_extension generated type drift test."""

from __future__ import annotations

from pathlib import Path

from scripts.codegen.generate_family_extension_types import (
    render_family_extension_types,
)


def test_family_extension_types_are_in_sync_with_shared_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    generated_path = repo_root / "apps/family_extension/src/contracts/generated.ts"

    expected = render_family_extension_types()
    actual = generated_path.read_text(encoding="utf-8")

    assert actual == expected
