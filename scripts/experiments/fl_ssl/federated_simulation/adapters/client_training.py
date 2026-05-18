"""FL simulation client local training adapter."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping
from typing import Any

from methods.adaptation.lora_classifier.config import (
    build_lora_classifier_training_backend_config,
)
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
from scripts.runtime_adapters.federated_agent.query_ssl_lora_classifier_trainer import (
    run_query_ssl_lora_classifier_local_training,
    upload_agent_local_lora_classifier_update,
)
from scripts.runtime_adapters.federated_agent.scoring_runtime import (
    build_federated_scoring_service,
)
from scripts.runtime_adapters.federated_agent.training_runtime import (
    run_federated_local_training,
)
from scripts.runtime_adapters.federated_server.runtime import SimulationServerRuntime
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_ADAPTER_KIND,
)
from shared.src.contracts.training_contracts import (
    ClientMetricKeys,
    TrainingUpdateEnvelope,
)

from ..io.selection_diagnostics_writer import SelectionDiagnosticsWriter


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

    if _should_run_query_ssl_lora_client_training(request):
        return _run_query_ssl_lora_client_round(
            request=request,
            bootstrapped=bootstrapped,
            active=active,
            round_id=round_id,
            shard=shard,
            training_task=training_task,
        )

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
    update_submitted = _accept_client_update(
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
            delta_l2_norm=_extract_delta_l2_norm(local_result.update_envelope),
            aggregation_example_count=_extract_aggregation_example_count(
                local_result.update_envelope
            ),
            client_train_time_seconds=client_train_time_seconds,
            client_payload_bytes=(
                _payload_byte_count(local_result.update_payload)
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


def _should_run_query_ssl_lora_client_training(
    request: SimulationRunRequest,
) -> bool:
    return (
        request.query_ssl_objective_config is not None
        and str(request.round_runtime_config.adapter_family_name).strip().lower()
        == LORA_CLASSIFIER_ADAPTER_KIND
        and request.round_runtime_config.lora_classifier is not None
    )


def _run_query_ssl_lora_client_round(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    round_id: str,
    shard: FederatedClientShard,
    training_task: Any,
) -> ClientRoundExecution:
    if request.query_ssl_objective_config is None:
        raise ValueError("query_ssl_objective_config is required.")
    if request.round_runtime_config.lora_classifier is None:
        raise ValueError("LoRA-classifier runtime config is required.")

    training_started_at = time.perf_counter()
    local_result = run_query_ssl_lora_classifier_local_training(
        client_id=shard.client_id,
        seed=request.seed,
        output_dir=request.output_dir,
        labeled_rows=shard.labeled_rows,
        unlabeled_rows=shard.unlabeled_rows,
        active_adapter_state=active.adapter_state,
        training_task=training_task,
        model_manifest=active.manifest,
        query_ssl_config=request.query_ssl_objective_config,
        lora_config=build_lora_classifier_training_backend_config(
            training_task.objective_config
        ),
        trainer_runtime_config=request.local_trainer_runtime_config,
    )
    client_train_time_seconds = time.perf_counter() - training_started_at
    server_update_payload = upload_agent_local_lora_classifier_update(
        output_dir=request.output_dir,
        update_payload=local_result.update_payload,
    )
    update_submitted = _accept_client_update(
        server_runtime=bootstrapped.server_runtime,
        round_id=round_id,
        update_envelope=local_result.update_envelope,
        update_payload=server_update_payload,
    )
    return ClientRoundExecution(
        summary=ClientRoundSummary(
            client_id=shard.client_id,
            candidate_count=local_result.candidate_count,
            accepted_count=local_result.accepted_count,
            update_generated=update_submitted,
            delta_l2_norm=_extract_delta_l2_norm(local_result.update_envelope),
            aggregation_example_count=_extract_aggregation_example_count(
                local_result.update_envelope
            ),
            client_train_time_seconds=client_train_time_seconds,
            client_payload_bytes=(
                _payload_byte_count(server_update_payload) if update_submitted else None
            ),
            pseudo_label_confidence_mean=local_result.client_metrics.get(
                ClientMetricKeys.MEAN_CONFIDENCE
            ),
            pseudo_label_margin_mean=local_result.client_metrics.get(
                ClientMetricKeys.MEAN_MARGIN
            ),
            pseudo_label_correct_count=0,
            pseudo_label_evaluated_count=0,
            accepted_label_distribution={},
            rejected_label_distribution={},
        ),
        update_submitted=update_submitted,
    )


def _accept_client_update(
    *,
    server_runtime: SimulationServerRuntime,
    round_id: str,
    update_envelope: TrainingUpdateEnvelope | None,
    update_payload: Any | None,
) -> bool:
    if update_envelope is None:
        return False
    if update_payload is None:
        raise ValueError("Update envelope exists without update payload.")
    server_runtime.accept_update(
        round_id,
        update_envelope,
        update_payload,
    )
    return True


def _payload_byte_count(update_payload: Any | None) -> int | None:
    if update_payload is None:
        return None
    if hasattr(update_payload, "model_dump_json"):
        return len(update_payload.model_dump_json().encode("utf-8"))
    if hasattr(update_payload, "model_dump"):
        update_payload = update_payload.model_dump(mode="json")
    return len(
        json.dumps(
            update_payload,
            default=str,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    )


def _extract_delta_l2_norm(
    update_envelope: TrainingUpdateEnvelope | None,
) -> float | None:
    if update_envelope is None:
        return None
    value = update_envelope.client_metrics.get(ClientMetricKeys.DELTA_L2_NORM)
    if value is None:
        return None
    return float(value)


def _extract_aggregation_example_count(
    update_envelope: TrainingUpdateEnvelope | None,
) -> int | None:
    if update_envelope is None:
        return None
    return int(update_envelope.example_count)


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
