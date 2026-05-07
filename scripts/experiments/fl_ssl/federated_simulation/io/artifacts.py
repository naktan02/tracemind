"""Federated simulation artifact writer compatibility facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientPoolSplitConfig,
    FederatedDatasetSplit,
    FederatedDiagnosticsConfig,
    FederatedReportConfig,
    FederatedRoundRuntimeConfig,
    FederatedSslMethodConfig,
    FederatedTrainingTaskConfig,
    FederatedValidationConfig,
    SimulationResult,
)
from scripts.io.labeled_query_rows import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.prototype_contracts import PrototypePackPayload

from .run_artifact_writer import RunArtifactWriter
from .selection_diagnostics_writer import SelectionDiagnosticsWriter
from .simulation_report_builder import SimulationReportBuilder


def save_selection_diagnostics(
    *,
    output_dir: Path,
    round_id: str,
    client_id: str,
    rows: list[LabeledQueryRow],
    training_examples: tuple[Any, ...],
    selection_result: Any,
    diagnostics_config: FederatedDiagnosticsConfig,
) -> tuple[Path, Path]:
    """row별 selection 원인과 요약을 저장한다."""

    return SelectionDiagnosticsWriter().save(
        output_dir=output_dir,
        round_id=round_id,
        client_id=client_id,
        rows=rows,
        training_examples=training_examples,
        selection_result=selection_result,
        diagnostics_config=diagnostics_config,
    )


def save_prototype_pack(output_dir: Path, payload: PrototypePackPayload) -> Path:
    """prototype pack payload를 output_dir 아래에 저장한다."""

    return RunArtifactWriter().save_prototype_pack(
        output_dir=output_dir,
        payload=payload,
    )


def save_model_manifest(output_dir: Path, manifest: ModelManifest) -> Path:
    """model manifest entity를 output_dir 아래 JSON으로 저장한다."""

    return RunArtifactWriter().save_model_manifest(
        output_dir=output_dir,
        manifest=manifest,
    )


def save_simulation_report(
    *,
    output_dir: Path,
    result: SimulationResult,
    report_config: FederatedReportConfig,
    client_count: int,
    round_budget: int,
    bootstrap_ratio: float,
    seed: int,
    shard_policy: FederatedShardPolicyConfig,
    dataset_split: FederatedDatasetSplit,
    ssl_method_config: FederatedSslMethodConfig,
    client_pool_split_config: FederatedClientPoolSplitConfig | None,
    training_task_config: FederatedTrainingTaskConfig,
    validation_config: FederatedValidationConfig,
    round_runtime_config: FederatedRoundRuntimeConfig,
) -> Path:
    """FL SSL main comparison 전용 report를 저장한다."""

    return SimulationReportBuilder().save(
        output_dir=output_dir,
        result=result,
        report_config=report_config,
        client_count=client_count,
        round_budget=round_budget,
        bootstrap_ratio=bootstrap_ratio,
        seed=seed,
        shard_policy=shard_policy,
        dataset_split=dataset_split,
        ssl_method_config=ssl_method_config,
        client_pool_split_config=client_pool_split_config,
        training_task_config=training_task_config,
        validation_config=validation_config,
        round_runtime_config=round_runtime_config,
    )
