"""FL simulation client local training adapter."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping
from typing import Any

from scripts.experiments.fl_ssl.federated_simulation.adapters.method_runtime import (
    FederatedSslSimulationRuntime,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
    BootstrappedSimulation,
    ClientRoundExecution,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientRoundSummary,
    FederatedClientShard,
    SimulationRunRequest,
)
from scripts.runtime_adapters.federated_agent.query_ssl_client_round import (
    run_query_ssl_client_round_if_supported,
)
from scripts.runtime_adapters.federated_agent.scoring_runtime import (
    build_federated_scoring_service,
)
from scripts.runtime_adapters.federated_agent.training_runtime import (
    run_federated_local_training,
)

from ..io.selection_diagnostics_writer import SelectionDiagnosticsWriter
from .client_update_submission import (
    accept_client_update,
    extract_aggregation_example_count,
    extract_delta_l2_norm,
    payload_byte_count,
)


def build_round_training_scoring_service(
    *,
    request: SimulationRunRequest,
    active: ActiveSimulationState,
    training_task: Any,
) -> Any:
    """현재 global state 기준 client local training scoring service를 만든다."""

    return build_federated_scoring_service(
        objective_config=training_task.objective_config,
        similarity_name=request.validation_config.similarity_name,
        shared_state=active.adapter_state,
    )


def run_client_round(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    ssl_method_runtime: FederatedSslSimulationRuntime,
    round_id: str,
    shard: FederatedClientShard,
    training_task: Any,
    training_scoring_service: Any,
) -> ClientRoundExecution:
    """client shard 하나의 local training을 실행하고 update를 제출한다."""

    query_ssl_execution = run_query_ssl_client_round_if_supported(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        round_id=round_id,
        shard=shard,
        training_task=training_task,
    )
    if query_ssl_execution is not None:
        return query_ssl_execution

    training_rows = ssl_method_runtime.select_training_rows(shard=shard)
    training_examples = ssl_method_runtime.build_training_examples(
        rows=training_rows,
        adapter=bootstrapped.adapter,
        adapter_state=active.adapter_state,
        prototype_pack=active.prototype_pack,
        model_id=request.model_id,
        scoring_service=training_scoring_service,
        objective_config=training_task.objective_config,
    )
    local_training_service = ssl_method_runtime.build_local_training_service(
        client_state_root=request.output_dir / "agents" / shard.client_id,
        training_task=training_task,
    )
    training_started_at = time.perf_counter()
    local_result = run_federated_local_training(
        local_training_service=local_training_service,
        training_examples=training_examples,
        training_task=training_task,
        model_manifest=active.manifest,
    )
    client_train_time_seconds = time.perf_counter() - training_started_at
    SelectionDiagnosticsWriter().save(
        output_dir=request.output_dir,
        round_id=round_id,
        client_id=shard.client_id,
        rows=training_rows,
        training_examples=training_examples,
        selection_result=local_result.selection_result,
        diagnostics_config=request.diagnostics_config,
    )
    update_submitted = accept_client_update(
        server_runtime=bootstrapped.server_runtime,
        round_id=round_id,
        update_envelope=local_result.update_envelope,
        update_payload=local_result.update_payload,
    )
    selection_quality = _build_selection_quality_summary(
        selection_result=local_result.selection_result,
        training_rows=training_rows,
    )
    return ClientRoundExecution(
        summary=ClientRoundSummary(
            client_id=shard.client_id,
            candidate_count=local_result.selection_result.total_count,
            accepted_count=local_result.selection_result.accepted_count,
            update_generated=update_submitted,
            delta_l2_norm=extract_delta_l2_norm(local_result.update_envelope),
            aggregation_example_count=extract_aggregation_example_count(
                local_result.update_envelope
            ),
            client_train_time_seconds=client_train_time_seconds,
            client_payload_bytes=(
                payload_byte_count(local_result.update_payload)
                if update_submitted
                else None
            ),
            pseudo_label_confidence_mean=selection_quality[
                "pseudo_label_confidence_mean"
            ],
            pseudo_label_margin_mean=selection_quality["pseudo_label_margin_mean"],
            pseudo_label_correct_count=selection_quality["pseudo_label_correct_count"],
            pseudo_label_evaluated_count=selection_quality[
                "pseudo_label_evaluated_count"
            ],
            accepted_label_distribution=selection_quality[
                "accepted_label_distribution"
            ],
            rejected_label_distribution=selection_quality[
                "rejected_label_distribution"
            ],
        ),
        update_submitted=update_submitted,
    )


def _build_selection_quality_summary(
    *,
    selection_result: Any,
    training_rows: list[Mapping[str, object]],
) -> dict[str, Any]:
    """simulation label을 아는 경우 accepted pseudo-label 품질을 요약한다."""

    candidates = tuple(selection_result.candidates)
    accepted_candidates = tuple(
        candidate for candidate in candidates if candidate.accepted
    )
    rejected_candidates = tuple(
        candidate for candidate in candidates if not candidate.accepted
    )
    true_label_by_query_id = {
        str(row["query_id"]): str(row["mapped_label_4"])
        for row in training_rows
        if "query_id" in row and "mapped_label_4" in row
    }
    evaluated_candidates = tuple(
        candidate
        for candidate in accepted_candidates
        if str(candidate.source_event_ref) in true_label_by_query_id
    )
    correct_count = sum(
        1
        for candidate in evaluated_candidates
        if true_label_by_query_id[str(candidate.source_event_ref)]
        == str(candidate.label)
    )
    return {
        "pseudo_label_confidence_mean": _mean_candidate_value(
            candidates,
            "confidence",
        ),
        "pseudo_label_margin_mean": _mean_candidate_value(candidates, "margin"),
        "pseudo_label_correct_count": correct_count,
        "pseudo_label_evaluated_count": len(evaluated_candidates),
        "accepted_label_distribution": _candidate_label_distribution(
            accepted_candidates
        ),
        "rejected_label_distribution": _candidate_label_distribution(
            rejected_candidates
        ),
    }


def _mean_candidate_value(
    candidates: tuple[Any, ...],
    field_name: str,
) -> float | None:
    values = [float(getattr(candidate, field_name)) for candidate in candidates]
    if not values:
        return None
    return sum(values) / len(values)


def _candidate_label_distribution(candidates: tuple[Any, ...]) -> dict[str, int]:
    return dict(
        sorted(Counter(str(candidate.label) for candidate in candidates).items())
    )
