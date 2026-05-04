"""Prototype rebuild publication 전략."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from methods.prototype.building.build_strategies import PrototypeBuildArtifacts
from shared.src.contracts.prototype_build_state_contracts import (
    dump_prototype_build_state_payload,
)
from shared.src.contracts.prototype_contracts import dump_prototype_pack_payload

from .models import (
    PrototypeRebuildResult,
)
from .prototype_build_state_service import (
    PrototypeBuildStateService,
)
from .prototype_pack_service import (
    PrototypePackService,
)


class PrototypePublicationStrategy(Protocol):
    """rebuild 결과물을 어디에 어떻게 발행할지 결정하는 전략 인터페이스."""

    def publish(
        self,
        build_artifacts: PrototypeBuildArtifacts,
    ) -> PrototypeRebuildResult:
        """빌드 결과물을 발행하고 경로를 반환한다."""


@dataclass(slots=True)
class InMemoryPrototypePublicationStrategy:
    """빌드 결과를 메모리에서만 반환하는 publication 전략."""

    def publish(
        self,
        build_artifacts: PrototypeBuildArtifacts,
    ) -> PrototypeRebuildResult:
        return PrototypeRebuildResult(
            pack_payload=build_artifacts.pack_payload,
            build_state_payload=build_artifacts.build_state_payload,
        )


@dataclass(slots=True)
class ReferenceRebuildPrototypePublicationStrategy:
    """reference rebuild 산출물을 local copy와 main_server state에 함께 발행한다."""

    reference_pack_output_dir: Path | None = None
    reference_build_state_output_dir: Path | None = None
    prototype_pack_service: PrototypePackService = field(
        default_factory=PrototypePackService
    )
    prototype_build_state_service: PrototypeBuildStateService = field(
        default_factory=PrototypeBuildStateService
    )

    def publish(
        self,
        build_artifacts: PrototypeBuildArtifacts,
    ) -> PrototypeRebuildResult:
        pack_payload = build_artifacts.pack_payload
        build_state_payload = build_artifacts.build_state_payload
        prototype_version = pack_payload.prototype_version

        reference_pack_path: Path | None = None
        if self.reference_pack_output_dir is not None:
            reference_pack_path = (
                self.reference_pack_output_dir / f"{prototype_version}.json"
            )
            dump_prototype_pack_payload(reference_pack_path, pack_payload)

        reference_build_state_path: Path | None = None
        if (
            self.reference_build_state_output_dir is not None
            and build_state_payload is not None
        ):
            reference_build_state_path = (
                self.reference_build_state_output_dir / f"{prototype_version}.json"
            )
            dump_prototype_build_state_payload(
                reference_build_state_path,
                build_state_payload,
            )

        published_build_state_path: Path | None = None
        if build_state_payload is not None:
            published_build_state_path = (
                self.prototype_build_state_service.publish_state(build_state_payload)
            )

        published_pack_path = self.prototype_pack_service.publish_pack(pack_payload)
        return PrototypeRebuildResult(
            pack_payload=pack_payload,
            build_state_payload=build_state_payload,
            published_pack_path=published_pack_path,
            published_build_state_path=published_build_state_path,
            reference_pack_path=reference_pack_path,
            reference_build_state_path=reference_build_state_path,
        )
