"""Federated simulation orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.prototype.building.base import PrototypeBuildStrategy
from scripts.experiments.fl_ssl.federated_simulation.artifacts import (
    save_model_manifest,
    save_prototype_pack,
    save_selection_diagnostics,
    save_simulation_report,
)
from scripts.experiments.fl_ssl.federated_simulation.evaluation import (
    build_validation_scoring_service,
    evaluate_rows,
)
from scripts.experiments.fl_ssl.federated_simulation.method_runtime import (
    build_federated_ssl_simulation_runtime,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientEvaluationSummary,
    ClientRoundSummary,
    FederatedDiagnosticsConfig,
    FederatedPrototypeRebuildConfig,
    FederatedReportConfig,
    FederatedRoundRuntimeConfig,
    FederatedSslMethodConfig,
    FederatedTrainingTaskConfig,
    FederatedValidationConfig,
    SimulationResult,
    SimulationRoundSummary,
)
from scripts.experiments.fl_ssl.federated_simulation.sharding import (
    split_rows_for_federation,
    split_rows_into_client_shards,
)
from scripts.labeled_query_rows import LabeledQueryRow
from scripts.runtime_adapters.embedding_runtime import create_embedding_adapter
from scripts.runtime_adapters.federated_agent_runtime import (
    build_federated_scoring_service,
    run_federated_local_training,
)
from scripts.runtime_adapters.federated_server_runtime import (
    SimulationServerRuntime,
    build_classifier_head_state_from_prototype_pack,
    build_initial_shared_state,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.prototype_contracts import load_prototype_pack_payload
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec


def run_simulation(
    *,
    train_rows: list[LabeledQueryRow],
    validation_rows: list[LabeledQueryRow],
    output_dir: Path,
    client_count: int,
    rounds: int,
    bootstrap_ratio: float,
    seed: int,
    embedding_spec: EmbeddingAdapterSpec,
    model_id: str,
    training_scope: str,
    round_runtime_config: FederatedRoundRuntimeConfig,
    prototype_build_strategy: PrototypeBuildStrategy,
    shard_policy: FederatedShardPolicyConfig,
    training_task_config: FederatedTrainingTaskConfig,
    validation_config: FederatedValidationConfig,
    prototype_rebuild_config: FederatedPrototypeRebuildConfig,
    diagnostics_config: FederatedDiagnosticsConfig,
    ssl_method_config: FederatedSslMethodConfig,
    report_config: FederatedReportConfig | None = None,
) -> SimulationResult:
    """bootstrap -> client pseudo-label -> aggregate -> republish 루프를 실행한다."""

    ssl_method_runtime = build_federated_ssl_simulation_runtime(ssl_method_config.name)
    ssl_method_descriptor = ssl_method_runtime.descriptor
    if ssl_method_config.implementation_status != (
        ssl_method_descriptor.implementation_status
    ):
        raise ValueError(
            "ssl_method implementation_status must match the registered "
            f"descriptor for {ssl_method_config.name}."
        )

    dataset_split = split_rows_for_federation(
        train_rows,
        bootstrap_ratio=bootstrap_ratio,
        client_count=client_count,
        seed=seed,
        shard_policy=shard_policy,
    )
    validation_client_shards = split_rows_into_client_shards(
        validation_rows,
        client_count=client_count,
        seed=seed + 1,
        shard_policy=shard_policy,
    )
    adapter = create_embedding_adapter(embedding_spec)
    if not dataset_split.bootstrap_rows:
        raise ValueError("Bootstrap split must contain at least one row.")
    embedding_dim = len(
        adapter.embed_texts([str(dataset_split.bootstrap_rows[0]["text"])])[0]
    )
    server_runtime = SimulationServerRuntime.build(
        output_dir=output_dir,
        round_runtime_config=round_runtime_config,
        prototype_build_strategy=prototype_build_strategy,
    )

    initial_model_revision = "sim_rev_0000"
    initial_prototype_version = "proto_sim_0000"
    now = datetime.now(timezone.utc)
    category_labels = tuple(
        sorted({str(row["mapped_label_4"]) for row in (*train_rows, *validation_rows)})
    )
    initial_state = build_initial_shared_state(
        adapter_family_name=round_runtime_config.adapter_family_name,
        model_id=model_id,
        model_revision=initial_model_revision,
        training_scope=training_scope,
        embedding_dim=embedding_dim,
        labels=category_labels,
        updated_at=now,
    )
    server_runtime.set_embedding_adapter(adapter)
    server_runtime.store_prototype_rebuild_input(
        rows=dataset_split.bootstrap_rows,
        embedding_spec=embedding_spec,
        rebuild_config=prototype_rebuild_config,
    )
    active_prototype = server_runtime.rebuild_reference_prototype_pack(
        adapter_state=initial_state,
        prototype_version=initial_prototype_version,
        embedding_model_id=model_id,
        embedding_model_revision=initial_model_revision,
        built_at=now,
    )
    if (
        round_runtime_config.adapter_family_name == "classifier_head"
        and getattr(prototype_build_strategy, "name", "") != "single"
    ):
        raise ValueError(
            "classifier_head bootstrap currently requires "
            "strategy_axes/prototype/build_strategy=single. "
            "The current bootstrap path converts one centroid per category into "
            "classifier weights and does not support multi-prototype packs yet."
        )
    if round_runtime_config.adapter_family_name == "classifier_head":
        initial_state = build_classifier_head_state_from_prototype_pack(
            prototype_pack=active_prototype,
            model_id=model_id,
            model_revision=initial_model_revision,
            training_scope=training_scope,
            updated_at=now,
            logit_scale=round_runtime_config.classifier_head_bootstrap_logit_scale,
        )
    validation_scoring_service = build_validation_scoring_service(
        validation_config,
        shared_state=initial_state,
    )
    initial_state_path = server_runtime.save_shared_adapter_state(initial_state)
    save_prototype_pack(output_dir, active_prototype)
    active_manifest = ModelManifest(
        schema_version="model_manifest.v1",
        model_id=model_id,
        model_revision=initial_model_revision,
        published_at=now,
        artifact_kind="shared_adapter_state",
        artifact_ref=str(initial_state_path),
        prototype_version=initial_prototype_version,
        training_scope=training_scope,
        training_enabled=True,
        compatible_task_types=("pseudo_label_self_training",),
        base_model_id=embedding_spec.model_id,
        base_model_revision=embedding_spec.revision,
        notes="round_active_pair_only bootstrap manifest",
    )
    save_model_manifest(output_dir, active_manifest)

    initial_validation = evaluate_rows(
        rows=validation_rows,
        adapter=adapter,
        adapter_state=initial_state,
        prototype_pack=active_prototype,
        model_id=model_id,
        scoring_service=validation_scoring_service,
        confidence_threshold=validation_config.confidence_threshold,
        margin_threshold=validation_config.margin_threshold,
        objective_config=training_task_config.objective_config,
    )

    round_summaries: list[SimulationRoundSummary] = []
    active_state = initial_state
    for round_index in range(1, rounds + 1):
        round_id = f"round_{round_index:04d}"
        round_record = server_runtime.open_round(
            ssl_method_runtime.build_round_open_request(
                active_manifest=active_manifest,
                round_id=round_id,
                training_task_config=training_task_config,
            )
        )
        training_task = round_record.training_task
        training_scoring_service = build_federated_scoring_service(
            objective_config=training_task.objective_config,
            similarity_name=validation_config.similarity_name,
            shared_state=active_state,
        )

        updates = []
        client_summaries: list[ClientRoundSummary] = []
        for shard in dataset_split.client_shards:
            training_examples = ssl_method_runtime.build_training_examples(
                rows=shard.rows,
                adapter=adapter,
                adapter_state=active_state,
                prototype_pack=active_prototype,
                model_id=model_id,
                scoring_service=training_scoring_service,
                objective_config=training_task.objective_config,
            )
            local_training_service = ssl_method_runtime.build_local_training_service(
                client_state_root=output_dir / "agents" / shard.client_id
            )
            local_result = run_federated_local_training(
                local_training_service=local_training_service,
                training_examples=training_examples,
                training_task=training_task,
                model_manifest=active_manifest,
            )
            save_selection_diagnostics(
                output_dir=output_dir,
                round_id=round_id,
                client_id=shard.client_id,
                rows=shard.rows,
                training_examples=training_examples,
                selection_result=local_result.selection_result,
                diagnostics_config=diagnostics_config,
            )
            if local_result.update_envelope is not None:
                server_runtime.accept_update(
                    round_id,
                    local_result.update_envelope,
                )
                updates.append(local_result.update_envelope)
            client_summaries.append(
                ClientRoundSummary(
                    client_id=shard.client_id,
                    candidate_count=local_result.selection_result.total_count,
                    accepted_count=local_result.selection_result.accepted_count,
                    update_generated=local_result.update_envelope is not None,
                )
            )

        if not updates:
            break

        next_model_revision = f"sim_rev_{round_index:04d}"
        next_prototype_version = f"proto_sim_{round_index:04d}"
        finalized_round = server_runtime.finalize_round(
            round_id=round_id,
            next_model_revision=next_model_revision,
            next_prototype_version=next_prototype_version,
        )
        if finalized_round.publication is None:
            raise ValueError("Finalized simulation round must contain publication.")
        active_manifest = finalized_round.publication.next_manifest
        active_state = server_runtime.load_active_state(active_manifest)
        save_model_manifest(output_dir, active_manifest)
        if finalized_round.publication.prototype_pack_ref is None:
            raise ValueError(
                "Simulation finalize must publish a prototype pack reference."
            )
        active_prototype = load_prototype_pack_payload(
            Path(finalized_round.publication.prototype_pack_ref)
        )
        save_prototype_pack(output_dir, active_prototype)
        validation = evaluate_rows(
            rows=validation_rows,
            adapter=adapter,
            adapter_state=active_state,
            prototype_pack=active_prototype,
            model_id=model_id,
            scoring_service=build_validation_scoring_service(
                validation_config,
                shared_state=active_state,
            ),
            confidence_threshold=validation_config.confidence_threshold,
            margin_threshold=validation_config.margin_threshold,
            objective_config=training_task.objective_config,
        )
        round_summaries.append(
            SimulationRoundSummary(
                round_id=round_id,
                model_revision=next_model_revision,
                prototype_version=next_prototype_version,
                update_count=len(updates),
                validation=validation,
                clients=tuple(client_summaries),
            )
        )

    final_validation = (
        round_summaries[-1].validation if round_summaries else initial_validation
    )
    final_validation_scoring_service = build_validation_scoring_service(
        validation_config,
        shared_state=active_state,
    )
    client_evaluations = tuple(
        ClientEvaluationSummary(
            client_id=shard.client_id,
            validation=evaluate_rows(
                rows=shard.rows,
                adapter=adapter,
                adapter_state=active_state,
                prototype_pack=active_prototype,
                model_id=model_id,
                scoring_service=final_validation_scoring_service,
                confidence_threshold=validation_config.confidence_threshold,
                margin_threshold=validation_config.margin_threshold,
                objective_config=training_task_config.objective_config,
            ),
        )
        for shard in validation_client_shards
    )
    result = SimulationResult(
        initial_model_revision=initial_model_revision,
        initial_prototype_version=initial_prototype_version,
        initial_validation=initial_validation,
        final_validation=final_validation,
        rounds=tuple(round_summaries),
        client_evaluations=client_evaluations,
    )
    if report_config is not None:
        result.report_path = str(
            save_simulation_report(
                output_dir=output_dir,
                result=result,
                report_config=report_config,
                client_count=client_count,
                round_budget=rounds,
                bootstrap_ratio=bootstrap_ratio,
                seed=seed,
                shard_policy=shard_policy,
                ssl_method_config=ssl_method_config,
                training_task_config=training_task_config,
                validation_config=validation_config,
                round_runtime_config=round_runtime_config,
            )
        )
    return result
