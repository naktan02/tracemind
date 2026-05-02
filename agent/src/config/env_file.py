"""간단한 `.env` 파일 로더.

외부 의존성 없이 agent-local `.env`와 repo root `.env`를 읽어 process 환경에 채운다.
이미 설정된 OS 환경변수는 덮어쓰지 않는다.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_ENV_PATH = REPO_ROOT / "agent" / ".env"
ROOT_ENV_PATH = REPO_ROOT / ".env"


def load_agent_env_files(
    *,
    environ: dict[str, str] | None = None,
) -> None:
    """agent-local `.env`를 우선 읽고 repo root `.env`를 fallback으로 읽는다."""

    target_environ = os.environ if environ is None else environ
    load_env_file(AGENT_ENV_PATH, environ=target_environ)
    load_env_file(ROOT_ENV_PATH, environ=target_environ)


def load_env_file(
    path: Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> None:
    """`.env` 파일을 읽어 비어 있는 환경변수만 채운다."""

    env_path = ROOT_ENV_PATH if path is None else path
    if not env_path.exists():
        return

    target_environ = os.environ if environ is None else environ
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        target_environ.setdefault(key, value)


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return key, _strip_optional_quotes(value.strip())


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


__all__ = [
    "AGENT_ENV_PATH",
    "ROOT_ENV_PATH",
    "load_agent_env_files",
    "load_env_file",
]
