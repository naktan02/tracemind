"""Agent local training backend registry facade.

Concrete local update backends live in `methods/adaptation/<family>/`.
Agent code keeps this facade only for runtime call sites that still import the
old path.
"""

from __future__ import annotations

from collections.abc import Callable

from methods.adaptation.local_update_backend import (
    SharedAdapterTrainingBackend,
    TrainingBackendFactory,
)
from methods.adaptation.local_update_registry import (
    build_shared_adapter_training_backend as _build_backend,
)
from methods.adaptation.local_update_registry import (
    list_registered_shared_adapter_training_backend_names as _list_names,
)
from methods.adaptation.local_update_registry import (
    list_shared_adapter_training_backend_catalog_entries as _list_catalog_entries,
)
from methods.adaptation.local_update_registry import (
    register_shared_adapter_training_backend as _register_backend,
)
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry


def register_shared_adapter_training_backend(
    *backend_names: str,
    catalog_entry: RegistryCatalogEntry,
    factory: TrainingBackendFactory | None = None,
) -> (
    Callable[[TrainingBackendFactory], TrainingBackendFactory] | TrainingBackendFactory
):
    """methods-owned local update backend registry에 factory를 등록한다."""

    return _register_backend(
        *backend_names,
        catalog_entry=catalog_entry,
        factory=factory,
    )


def build_shared_adapter_training_backend(
    backend_name: str,
    *,
    objective_config=None,
) -> SharedAdapterTrainingBackend:
    """backend 이름으로 local update backend를 생성한다."""

    return _build_backend(backend_name, objective_config=objective_config)


def list_registered_shared_adapter_training_backend_names() -> tuple[str, ...]:
    """등록된 local update backend 이름을 정렬된 tuple로 반환한다."""

    return _list_names()


def list_shared_adapter_training_backend_catalog_entries() -> tuple[
    RegistryCatalogEntry, ...
]:
    """등록된 local update backend catalog entry를 canonical item 기준으로 반환한다."""

    return _list_catalog_entries()
