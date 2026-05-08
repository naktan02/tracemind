"""Server aggregation artifact reference helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AggregatedArtifactRefBuilder:
    """next revision 기준 server-owned artifact ref를 만든다."""

    artifact_ref_prefix: str
    artifact_format: str

    def __post_init__(self) -> None:
        _require_non_empty_str(
            self.artifact_ref_prefix,
            field_name="artifact_ref_prefix",
        )
        _require_non_empty_str(self.artifact_format, field_name="artifact_format")

    def build_ref(
        self,
        *,
        next_model_revision: str,
        artifact_name: str,
    ) -> str:
        """prefix/revision/name으로 opaque server-owned ref를 만든다."""

        return "/".join(
            (
                self.artifact_ref_prefix.rstrip("/"),
                _slug_ref_part(next_model_revision),
                _slug_ref_part(artifact_name),
            )
        )


def _slug_ref_part(value: str) -> str:
    normalized = value.strip().replace("/", "_")
    if not normalized:
        raise ValueError("artifact ref path parts must not be empty.")
    return normalized


def _require_non_empty_str(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty.")
