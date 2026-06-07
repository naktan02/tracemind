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

    from main_server.src.services.federation.prototypes import (
        prototype_rebuild_service as rebuild_service_module,
    )
    from main_server.src.services.federation.prototypes import (
        publication_strategies as publication_strategy_module,
    )

    return rebuild_service_module.PrototypeRebuildService(
        build_strategy=build_strategy,
        publication_strategy=(
            publication_strategy_module.ReferenceRebuildPrototypePublicationStrategy(
                reference_pack_output_dir=output_dir,
                reference_build_state_output_dir=build_state_output_dir,
            )
        ),
    )


def publish_prototype_build_state(payload: Any) -> Path:
    """main_server state 저장소에 prototype build state를 publish한다."""

    from main_server.src.services.federation.prototypes import (
        prototype_build_state_service as build_state_service_module,
    )

    return build_state_service_module.PrototypeBuildStateService().publish_state(
        payload
    )


def publish_prototype_pack(payload: Any) -> Path:
    """main_server state 저장소에 prototype pack을 publish한다."""

    from main_server.src.services.federation.prototypes import (
        prototype_pack_service as pack_service_module,
    )

    return pack_service_module.PrototypePackService().publish_pack(payload)


def activate_prototype_pack_version(prototype_version: str) -> Any:
    """main_server state 저장소의 활성 PrototypePack 버전을 변경한다."""

    from main_server.src.services.federation.prototypes import (
        prototype_pack_service as pack_service_module,
    )

    return pack_service_module.PrototypePackService().activate(prototype_version)
