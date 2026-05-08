"""Server aggregation artifact reference helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAIN_SERVER_ROOT = Path(__file__).resolve().parents[5]
AGGREGATION_ARTIFACT_REF_PREFIX = "aggregation_artifact::"


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


@dataclass(slots=True)
class AggregationArtifactStore:
    """server-owned aggregation artifact ref를 JSON artifact로 materialize한다."""

    state_root: Path = MAIN_SERVER_ROOT / "state" / "aggregation_artifacts"

    @property
    def artifacts_dir(self) -> Path:
        return self.state_root / "versions"

    def ref_for_artifact(self, artifact_id: str) -> str:
        """test/future upload path가 사용할 opaque server-owned artifact ref."""

        return (
            f"{AGGREGATION_ARTIFACT_REF_PREFIX}"
            f"{'/'.join(_safe_artifact_id_parts(artifact_id))}"
        )

    def path_for_artifact(self, artifact_id: str) -> Path:
        """artifact_id를 repository 내부 JSON path로 변환한다."""

        parts = _safe_artifact_id_parts(artifact_id)
        leaf = f"{parts[-1]}.json"
        return self.artifacts_dir.joinpath(*parts[:-1], leaf)

    def save_json_artifact(
        self,
        artifact_id: str,
        payload: dict[str, Any],
    ) -> Path:
        """server-owned artifact JSON을 저장한다."""

        path = self.path_for_artifact(artifact_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return path

    def load_json_artifact(
        self,
        *,
        artifact_ref: str,
    ) -> dict[str, object]:
        """server-owned artifact ref를 JSON object로 읽는다."""

        artifact_id = self.artifact_id_from_ref(artifact_ref)
        if artifact_id is None:
            raise FileNotFoundError(
                "Unsupported aggregation artifact ref. Expected server-owned "
                f"{AGGREGATION_ARTIFACT_REF_PREFIX!r} ref: {artifact_ref}"
            )
        path = self.path_for_artifact(artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"Aggregation artifact not found: {artifact_ref}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(
                f"Aggregation artifact must be a JSON object: {artifact_ref}"
            )
        return payload

    @staticmethod
    def artifact_id_from_ref(artifact_ref: str) -> str | None:
        """opaque artifact ref에서 repository-local artifact id를 추출한다."""

        if not artifact_ref.startswith(AGGREGATION_ARTIFACT_REF_PREFIX):
            return None
        artifact_id = artifact_ref.removeprefix(AGGREGATION_ARTIFACT_REF_PREFIX)
        return "/".join(_safe_artifact_id_parts(artifact_id))


def _slug_ref_part(value: str) -> str:
    normalized = value.strip().replace("/", "_")
    if not normalized:
        raise ValueError("artifact ref path parts must not be empty.")
    return normalized


def _require_non_empty_str(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty.")


def _safe_artifact_id_parts(artifact_id: str) -> tuple[str, ...]:
    parts = tuple(
        part.strip() for part in artifact_id.strip().split("/") if part.strip()
    )
    if not parts:
        raise ValueError("artifact_id must not be empty.")
    if any(part in {".", ".."} for part in parts):
        raise ValueError("artifact_id must not contain path traversal parts.")
    return parts
