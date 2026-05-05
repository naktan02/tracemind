"""experiment_web generated type drift test."""

from __future__ import annotations

from pathlib import Path

from scripts.codegen.generate_experiment_web_types import render_experiment_web_types


def test_experiment_web_types_are_in_sync_with_backend_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    generated_path = repo_root / "apps/experiment_web/src/types.ts"

    expected = render_experiment_web_types()
    actual = generated_path.read_text(encoding="utf-8")

    assert actual == expected
