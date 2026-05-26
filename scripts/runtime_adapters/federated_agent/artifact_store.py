"""Federated-agent simulation artifact store bridge."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregationArtifactStore,
)
from shared.src.contracts.training_contracts import TrainingTask

AGENT_LOCAL_ARTIFACT_REF_PREFIX = "agent-local://"


@dataclass(frozen=True, slots=True)
class SimulationClientArtifactStore:
    """simulation output directory를 client/server artifact 저장소로 연결한다."""

    output_dir: Path

    def ref_for_agent_artifact(
        self,
        *,
        artifact_ref_prefix: str,
        training_task: TrainingTask,
        client_id: str,
        update_id: str,
        artifact_name: str,
    ) -> str:
        normalized_prefix = artifact_ref_prefix.strip().rstrip("/")
        if not normalized_prefix.startswith(AGENT_LOCAL_ARTIFACT_REF_PREFIX):
            raise ValueError(
                "agent-local artifact_ref_prefix must start with "
                f"{AGENT_LOCAL_ARTIFACT_REF_PREFIX!r}."
            )
        return "/".join(
            (
                normalized_prefix,
                _safe_ref_part(training_task.round_id),
                _safe_ref_part(client_id),
                _safe_ref_part(update_id),
                _safe_ref_part(artifact_name),
            )
        )

    def save_agent_json_artifact(
        self,
        *,
        artifact_ref: str,
        payload: Mapping[str, object],
    ) -> None:
        path = self._path_for_agent_local_artifact(artifact_ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(dict(payload), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    def ref_for_server_client_update_artifact(
        self,
        *,
        training_task: TrainingTask,
        client_id: str,
        update_id: str,
        artifact_name: str,
    ) -> str:
        artifact_id_prefix = (
            f"client_updates/{training_task.round_id}/{client_id}/{update_id}"
        )
        return self._server_store().ref_for_artifact(
            f"{artifact_id_prefix}/{artifact_name}"
        )

    def save_server_safetensors_artifact_ref(
        self,
        *,
        artifact_ref: str,
        tensors: Mapping[str, object],
        metadata: Mapping[str, str],
    ) -> None:
        self._server_store().save_safetensors_artifact_ref(
            artifact_ref=artifact_ref,
            tensors=tensors,
            metadata=metadata,
        )

    def is_agent_local_ref(self, artifact_ref: str | None) -> bool:
        return artifact_ref is not None and artifact_ref.startswith(
            AGENT_LOCAL_ARTIFACT_REF_PREFIX
        )

    def upload_agent_local_json_artifact(self, *, agent_local_ref: str) -> str:
        payload = self._load_agent_local_json_artifact(agent_local_ref)
        store = self._server_store()
        server_ref = store.ref_for_artifact(
            "client_uploads/" + "/".join(self._agent_local_artifact_parts(agent_local_ref))
        )
        store.save_json_artifact_ref(artifact_ref=server_ref, payload=payload)
        return server_ref

    def server_artifact_refs_byte_count(
        self,
        *,
        artifact_refs: Sequence[str | None],
    ) -> int:
        store = self._server_store()
        total = 0
        for artifact_ref in artifact_refs:
            if artifact_ref is None:
                continue
            artifact_id = store.artifact_id_from_ref(artifact_ref)
            if artifact_id is None:
                continue
            for path in (
                store.path_for_artifact(artifact_id),
                store.path_for_safetensors_artifact(artifact_id),
            ):
                if path.exists():
                    total += path.stat().st_size
        return total

    def _server_store(self) -> AggregationArtifactStore:
        return AggregationArtifactStore(
            state_root=self.output_dir / "main_server" / "aggregation_artifacts"
        )

    def _load_agent_local_json_artifact(self, artifact_ref: str) -> dict[str, object]:
        path = self._path_for_agent_local_artifact(artifact_ref)
        if not path.exists():
            raise FileNotFoundError(f"Agent-local artifact not found: {artifact_ref}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(
                f"Agent-local artifact must be a JSON object: {artifact_ref}"
            )
        return payload

    def _path_for_agent_local_artifact(self, artifact_ref: str) -> Path:
        parts = self._agent_local_artifact_parts(artifact_ref)
        return (
            self.output_dir
            / "agents"
            / "local_artifacts"
            / "versions"
            / Path(*parts[:-1])
            / f"{parts[-1]}.json"
        )

    def _agent_local_artifact_parts(self, artifact_ref: str) -> tuple[str, ...]:
        if not artifact_ref.startswith(AGENT_LOCAL_ARTIFACT_REF_PREFIX):
            raise ValueError(f"Expected agent-local artifact ref: {artifact_ref!r}.")
        raw_artifact_id = artifact_ref.removeprefix(AGENT_LOCAL_ARTIFACT_REF_PREFIX)
        parts = tuple(part.strip() for part in raw_artifact_id.split("/") if part.strip())
        if not parts:
            raise ValueError("agent-local artifact ref must contain an artifact id.")
        if any(part in {".", ".."} for part in parts):
            raise ValueError("agent-local artifact ref must not contain path traversal.")
        return parts


def _safe_ref_part(value: str) -> str:
    normalized = str(value).strip().replace("/", "_")
    if not normalized or normalized in {".", ".."}:
        raise ValueError("artifact ref path parts must not be empty or traversal.")
    return normalized
