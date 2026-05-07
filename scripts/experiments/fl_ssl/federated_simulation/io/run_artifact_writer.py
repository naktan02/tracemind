"""FL simulation run artifact writer."""

from __future__ import annotations

from pathlib import Path

from shared.src.contracts.model_contracts import (
    ModelManifest,
    dump_model_manifest_payload,
)
from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    dump_prototype_pack_payload,
)


class RunArtifactWriter:
    """Run 단위 manifest/prototype artifact 저장 위치를 소유한다."""

    def save_prototype_pack(
        self,
        *,
        output_dir: Path,
        payload: PrototypePackPayload,
    ) -> Path:
        path = (
            output_dir
            / "main_server"
            / "prototype_packs"
            / f"{payload.prototype_version}.json"
        )
        dump_prototype_pack_payload(path, payload)
        return path

    def save_model_manifest(
        self,
        *,
        output_dir: Path,
        manifest: ModelManifest,
    ) -> Path:
        path = (
            output_dir
            / "main_server"
            / "model_manifests"
            / f"{manifest.model_revision}.json"
        )
        dump_model_manifest_payload(path, manifest)
        return path
