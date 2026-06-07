"""Training API unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent.src.api import training as training_api
from agent.src.services.training.execution.agent_training_task_runner_service import (
    AgentTrainingTaskRunRequest,
    AgentTrainingTaskRunResult,
)


def test_run_current_task_delegates_to_runner_service() -> None:
    runner_service = MagicMock()
    runner_service.run_current_task.return_value = AgentTrainingTaskRunResult(
        status="uploaded",
        round_id="round_0001",
        task_id="task_001",
        update_id="update_001",
        example_count=3,
        accepted_count=2,
        message="ok",
    )

    response = training_api.run_current_task(
        training_api.RunCurrentTaskRequest(
            server_base_url="http://server.test",
            analysis_event_days=14,
            agent_id="agent_a",
        ),
        runner_service=runner_service,
    )

    runner_service.run_current_task.assert_called_once_with(
        AgentTrainingTaskRunRequest(
            server_base_url="http://server.test",
            analysis_event_days=14,
            agent_id="agent_a",
        )
    )
    assert response.status == "uploaded"
    assert response.round_id == "round_0001"
    assert response.task_id == "task_001"
    assert response.update_id == "update_001"
    assert response.example_count == 3
    assert response.accepted_count == 2


def test_run_current_task_request_accepts_debug_page_camel_case_url() -> None:
    request = training_api.RunCurrentTaskRequest.model_validate(
        {
            "serverBaseUrl": "http://server.test",
            "analysis_event_days": 7,
        }
    )

    assert request.server_base_url == "http://server.test"
    assert request.analysis_event_days == 7
