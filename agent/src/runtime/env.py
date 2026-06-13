"""Agent runtime 환경값 정규화."""

from __future__ import annotations

import os
from collections.abc import Mapping

FAMILY_EXTENSION_ALLOWED_ORIGINS_ENV = "FAMILY_EXTENSION_ALLOWED_ORIGINS"
DEFAULT_FAMILY_EXTENSION_ALLOWED_ORIGINS = (
    "http://localhost:5174",
    "http://127.0.0.1:5174",
)


def load_family_extension_allowed_origins_from_env(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """family_extension dev server가 접근할 수 있는 origin 목록을 읽는다."""

    effective_environ = os.environ if environ is None else environ
    raw_value = effective_environ.get(FAMILY_EXTENSION_ALLOWED_ORIGINS_ENV, "")
    origins = tuple(origin.strip() for origin in raw_value.split(",") if origin.strip())
    return origins or DEFAULT_FAMILY_EXTENSION_ALLOWED_ORIGINS
