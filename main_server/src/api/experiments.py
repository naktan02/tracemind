"""개발자용 experiment catalog API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from main_server.src.services.experiments.catalog_service import (
    ExperimentCatalogService,
)
from main_server.src.services.experiments.payloads import ExperimentCatalogPayload

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


def get_experiment_catalog_service(request: Request) -> ExperimentCatalogService:
    """app.state에서 ExperimentCatalogService를 읽는다."""

    service = getattr(request.app.state, "experiment_catalog_service", None)
    if service is None:
        raise RuntimeError(
            "ExperimentCatalogService가 app.state에 설정되지 않았습니다. "
            "앱 생성 시 app.state.experiment_catalog_service를 설정하세요."
        )
    return service


ExperimentCatalogServiceDep = Annotated[
    ExperimentCatalogService,
    Depends(get_experiment_catalog_service),
]


@router.get(
    "/catalog",
    response_model=ExperimentCatalogPayload,
)
def get_experiment_catalog(
    service: ExperimentCatalogServiceDep,
) -> ExperimentCatalogPayload:
    """현재 코드/설정 기준 read-only experiment catalog를 반환한다."""

    return service.build_catalog()
