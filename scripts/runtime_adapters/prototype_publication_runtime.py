"""prototype pack publish/activation CLI bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from methods.prototype.building.base import PrototypeBuildStrategy


def build_reference_prototype_rebuild_service(
    *,
    output_dir: Path,
    build_state_output_dir: Path,
    build_strategy: PrototypeBuildStrategy,
) -> Any:
    """reference pack과 build state를 기록하는 rebuild service를 만든다."""

    from main_server.src.services.federation.assets.prototypes import (
        PrototypeRebuildService,
        ReferenceRebuildPrototypePublicationStrategy,
    )

    return PrototypeRebuildService(
        build_strategy=build_strategy,
        publication_strategy=ReferenceRebuildPrototypePublicationStrategy(
            reference_pack_output_dir=output_dir,
            reference_build_state_output_dir=build_state_output_dir,
        ),
    )


def publish_prototype_build_state(payload: Any) -> Path:
    """main_server state 저장소에 prototype build state를 publish한다."""

    from main_server.src.services.federation.assets.prototypes import (
        PrototypeBuildStateService,
    )

    return PrototypeBuildStateService().publish_state(payload)


def publish_prototype_pack(payload: Any) -> Path:
    """main_server state 저장소에 prototype pack을 publish한다."""

    from main_server.src.services.federation.assets.prototypes import (
        PrototypePackService,
    )

    return PrototypePackService().publish_pack(payload)


def activate_prototype_pack_version(prototype_version: str) -> Any:
    """main_server state 저장소의 활성 PrototypePack 버전을 변경한다."""

    from main_server.src.services.federation.assets.prototypes import (
        PrototypePackService,
    )

    return PrototypePackService().activate(prototype_version)


def pull_current_prototype_pack(server_base_url: str) -> Any:
    """agent local cache로 현재 활성 PrototypePack을 내려받는다."""

    from agent.src.services.assets.prototypes.sync_service import PrototypeSyncService

    return PrototypeSyncService().pull_current(server_base_url=server_base_url)
