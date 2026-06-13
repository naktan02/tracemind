"""Agent runtime filesystem paths."""

from __future__ import annotations

from pathlib import Path

AGENT_SRC_DIR = Path(__file__).resolve().parents[1]
AGENT_ROOT_DIR = AGENT_SRC_DIR.parent
DEFAULT_AGENT_DATA_DIR = AGENT_ROOT_DIR / "data"
DEFAULT_AGENT_STATE_DIR = AGENT_ROOT_DIR / "state"
