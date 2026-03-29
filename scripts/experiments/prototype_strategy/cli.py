"""Prototype strategy 실험 CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

@dataclass(slots=True, frozen=True)
class CliArgs:
    config_path: Path
    overrides: tuple[str, ...]


def parse_cli_args(*, default_config_path: Path) -> CliArgs:
    """얇은 CLI: config 파일과 override만 받는다."""
    parser = argparse.ArgumentParser(
        description=(
            "Run prototype strategy experiment from a YAML config. "
            "Use --override key=value to override config fields."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path,
        help="Path to the prototype strategy YAML config.",
    )
    parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        default=[],
        help=(
            "Override config values with dotlist syntax. "
            "Example: --override embedding.device=cuda"
        ),
    )
    args = parser.parse_args()
    return CliArgs(
        config_path=args.config,
        overrides=tuple(args.overrides),
    )
