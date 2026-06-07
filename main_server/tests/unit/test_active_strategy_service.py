"""active strategy 전환 서비스/API 테스트."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from main_server.src.api.main import create_app
from main_server.src.infrastructure.repositories.active_strategy_repository import (
    ActiveStrategyRepository,
)
from main_server.src.services.federation.strategy.active_strategy_service import (
    ActiveStrategyService,
)
from shared.src.domain.services.clock import FixedClock


def _build_service(tmp_path: Path) -> ActiveStrategyService:
    return ActiveStrategyService(
        repository=ActiveStrategyRepository(state_root=tmp_path / "active_strategy"),
        clock=FixedClock(datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)),
    )


def test_active_strategy_service_saves_current_and_history(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    config = service.switch(
        ssl_method="flexmatch_usb_v1",
        aggregation_backend="fedavg",
        notes="switch to FlexMatch",
    )

    assert config.ssl_method == "flexmatch_usb_v1"
    assert service.get_active_strategy().ssl_method == "flexmatch_usb_v1"
    history = service.get_history()
    assert len(history) == 1
    assert history[0].notes == "switch to FlexMatch"


def test_active_strategy_api_switches_current_strategy(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    app = create_app(active_strategy_service=service)

    with TestClient(app, base_url="http://testserver") as client:
        response = client.post(
            "/api/v1/admin/strategy",
            json={
                "ssl_method": "flexmatch_usb_v1",
                "aggregation_backend": "fedavg",
                "notes": "api switch",
            },
        )
        current_response = client.get("/api/v1/admin/strategy/current")

    assert response.status_code == 200
    assert response.json()["ssl_method"] == "flexmatch_usb_v1"
    assert current_response.status_code == 200
    assert current_response.json()["ssl_method"] == "flexmatch_usb_v1"


def test_active_strategy_api_switches_live_server_fssl_method(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    app = create_app(active_strategy_service=service)

    with TestClient(app, base_url="http://testserver") as client:
        response = client.post(
            "/api/v1/admin/strategy",
            json={
                "ssl_method": "fixmatch_usb_v1",
                "fssl_method": "fedmatch",
                "aggregation_backend": "fedavg",
            },
        )

    assert response.status_code == 200
    assert response.json()["fssl_method"] == "fedmatch"
    assert service.get_active_strategy().fssl_method == "fedmatch"


def test_active_strategy_api_clears_fssl_method(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    service.switch(fssl_method="fedmatch")
    app = create_app(active_strategy_service=service)

    with TestClient(app, base_url="http://testserver") as client:
        response = client.post(
            "/api/v1/admin/strategy",
            json={"fssl_method": ""},
        )

    assert response.status_code == 200
    assert response.json()["fssl_method"] is None
    assert service.get_active_strategy().fssl_method is None


def test_active_strategy_api_rejects_unknown_ssl_method(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    app = create_app(active_strategy_service=service)

    with TestClient(app, base_url="http://testserver") as client:
        response = client.post(
            "/api/v1/admin/strategy",
            json={"ssl_method": "unknown_ssl"},
        )

    assert response.status_code == 400
    assert "지원되지 않는 ssl_method" in response.json()["detail"]
