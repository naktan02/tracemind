"""FL simulation runtime adapter 아키텍처 guard."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_SRC = REPO_ROOT / "scripts"
SCRIPTS_RUNTIME_ADAPTER_SRC = SCRIPTS_SRC / "runtime_adapters"
FL_SIMULATION_IO_SRC = (
    SCRIPTS_SRC / "experiments" / "fl_ssl" / "federated_simulation" / "io"
)


def _relative_repo_path(path: Path) -> Path:
    return path.relative_to(REPO_ROOT)


def test_fl_simulation_io_does_not_keep_artifact_facade() -> None:
    facade_path = FL_SIMULATION_IO_SRC / "artifacts.py"

    assert not facade_path.exists(), (
        "FL simulation artifact I/O는 중앙 artifacts.py facade 없이 writer/builder를 "
        "직접 호출한다. facade가 필요해 보이면 builder/writer 책임이 얕은지 "
        "먼저 점검한다.\n"
        f"facade path={_relative_repo_path(facade_path)}"
    )


def test_fl_simulation_does_not_keep_task_config_facade() -> None:
    facade_path = (
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "adapters"
        / "task_config.py"
    )

    assert not facade_path.exists(), (
        "FL simulation task config 변환은 runtime bridge의 task_config_surface.py와 "
        "round_request_mapper.py에 둔다. adapters/task_config.py pass-through facade를 "
        "다시 만들지 않는다.\n"
        f"facade path={_relative_repo_path(facade_path)}"
    )


def test_fl_simulation_client_training_has_no_payload_adapter_literals() -> None:
    path = (
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "adapters"
        / "client_training.py"
    )
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "classifier_head",
        "diagonal_scale",
        "lora_classifier",
        "ClassifierHead",
        "DiagonalScale",
        "LoraClassifier",
        "LORA_CLASSIFIER",
        "LoRA-classifier",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "client_training.py는 client round orchestration만 맡고 payload-adapter별 "
        "raw-row training, artifact upload, payload 변환은 federated_agent runtime "
        "adapter로 낮춘다.\n"
        f"violations={violations}"
    )


def test_fl_simulation_config_callable_loading_is_centralized() -> None:
    callable_loader_path = (
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "adapters"
        / "runtime_callable_loader.py"
    )
    checked_paths = (
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "config_request.py",
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "adapters"
        / "local_objective_execution.py",
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "adapters"
        / "server_step_execution.py",
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "adapters"
        / "evaluation.py",
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "flow"
        / "result_builder.py",
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "io"
        / "final_projection.py",
    )
    forbidden_snippet = "from importlib import import_module"
    violations = [
        _relative_repo_path(path)
        for path in checked_paths
        if forbidden_snippet in path.read_text(encoding="utf-8")
    ]

    assert callable_loader_path.exists(), (
        "round_runtime이 선언한 callable import/validation은 한 helper가 소유한다. "
        f"missing={_relative_repo_path(callable_loader_path)}"
    )
    assert not violations, (
        "FL simulation adapter들은 config-declared callable을 실행만 하고, "
        "fully-qualified path 파싱과 import 규칙은 runtime_callable_loader.py에 "
        "모은다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_fl_single_simulation_entrypoint_stays_thin() -> None:
    path = SCRIPTS_SRC / "experiments" / "fl_ssl" / "run_federated_simulation.py"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "def _build_training_task_config",
        "def _resolve_fl_data_source",
        "load_materialized_client_split",
        "LocalUpdateProfile",
        "TrainingObjectiveConfig",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "run_federated_simulation.py는 Hydra entrypoint, run dir, result line 출력만 "
        "맡는다. config-to-SimulationRunRequest 해석은 "
        "federated_simulation/config_request.py에 둔다.\n"
        f"violations={violations}"
    )


def test_fl_simulation_public_api_uses_typed_request_only() -> None:
    path = (
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "simulation.py"
    )
    source = path.read_text(encoding="utf-8")

    assert "def run_simulation(" not in source, (
        "FL simulation public API는 typed SimulationRunRequest를 받는 "
        "run_simulation_request만 둔다. 넓은 keyword wrapper는 caller가 전체 "
        "조립 세부사항을 반복하게 하므로 재도입하지 않는다."
    )


def test_federated_ssl_simulation_runtime_keeps_round_open_surface_only() -> None:
    path = (
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "adapters"
        / "method_runtime.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    runtime_class = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and node.name == "FederatedSslSimulationRuntime"
        ),
        None,
    )
    assert runtime_class is not None
    method_names = {
        node.name for node in runtime_class.body if isinstance(node, ast.FunctionDef)
    }

    assert method_names == {"build_round_open_request"}, (
        "FederatedSslSimulationRuntime caller surface는 round open seam만 유지한다. "
        "local training plan은 prototype-scored fallback 제거 후 실제 caller가 없는 "
        "future seam이므로 protocol에 다시 노출하지 않는다.\n"
        f"methods={sorted(method_names)}"
    )


def test_fl_simulation_report_builder_does_not_write_report_json() -> None:
    builder_path = FL_SIMULATION_IO_SRC / "simulation_report_builder.py"
    writer_path = FL_SIMULATION_IO_SRC / "simulation_report_writer.py"
    source = builder_path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "json.dumps(",
        ".write_text(",
        ".mkdir(",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert writer_path.exists(), (
        "FL simulation report는 payload builder와 JSON writer를 분리한다. "
        f"missing writer={_relative_repo_path(writer_path)}"
    )
    assert not violations, (
        "SimulationReportBuilder는 schema payload만 조립한다. report path, directory, "
        "JSON serialization, 파일 write는 SimulationReportWriter가 맡는다.\n"
        f"violations={violations}"
    )


def test_scripts_runtime_adapters_do_not_keep_federated_server_facade() -> None:
    facade_path = SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_server_runtime.py"
    family_state_bridge_path = (
        SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_server" / "lora_classifier_state.py"
    )

    assert not facade_path.exists(), (
        "FL simulation server runtime adapter는 federated_server/ package의 책임별 "
        "module을 직접 import한다. 중앙 re-export facade를 다시 만들지 않는다.\n"
        f"facade path={_relative_repo_path(facade_path)}"
    )
    assert not family_state_bridge_path.exists(), (
        "scripts federated_server bridge는 payload-adapter state publication 파일을 "
        "소유하지 않는다. projection 의미는 methods/adaptation/<family>/가, "
        "server-owned 저장/activate mechanism은 main_server publication capability가 "
        "소유한다.\n"
        f"path={_relative_repo_path(family_state_bridge_path)}"
    )


def test_federated_server_peft_runtime_adapter_keeps_method_core_in_methods() -> None:
    server_step_path = (
        SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_server" / "peft_encoder_server_step.py"
    )
    final_projection_path = (
        SCRIPTS_RUNTIME_ADAPTER_SRC
        / "federated_server"
        / "peft_encoder_final_projection.py"
    )
    method_runtime_paths = (
        REPO_ROOT
        / "methods"
        / "adaptation"
        / "peft_text_encoder"
        / "simulation_runtime"
        / "supervised_seed.py",
        REPO_ROOT
        / "methods"
        / "adaptation"
        / "peft_text_encoder"
        / "simulation_runtime"
        / "final_projection.py",
    )
    forbidden_snippets = (
        "run_peft_encoder_supervised_seed_step_core",
        "build_peft_encoder_state_projection",
        "write_peft_encoder_projection_artifacts",
        "build_peft_encoder_text_classifier_from_config",
        "load_peft_encoder_base_parameters_into_model",
        "materialize_base_peft_encoder_state",
        "build_dataloader",
    )
    violations: list[tuple[Path, str]] = []
    for path in (server_step_path, final_projection_path):
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in source:
                violations.append((_relative_repo_path(path), snippet))
    missing_method_paths = [
        _relative_repo_path(path) for path in method_runtime_paths if not path.exists()
    ]

    assert not missing_method_paths, (
        "PEFT encoder simulation server core는 methods/adaptation/<family>/"
        "simulation_runtime/가 소유한다.\n"
        f"{chr(10).join(f'- {path}' for path in missing_method_paths)}"
    )
    assert not violations, (
        "scripts federated_server PEFT adapter는 request/context/publication bridge만 "
        "맡는다. 모델 빌드, state materialization, seed step, projection assembly는 "
        "methods/adaptation/peft_text_encoder/simulation_runtime/로 둔다.\n"
        f"{chr(10).join(f'- {path}: {snippet}' for path, snippet in violations)}"
    )


def test_scripts_runtime_adapters_do_not_keep_federated_agent_family_files() -> None:
    package_root = SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_agent"
    forbidden_paths = (
        package_root / "lora_classifier_artifacts.py",
        package_root / "lora_classifier_base_state.py",
        package_root / "method_owned_lora_classifier_trainer.py",
        package_root / "query_ssl_lora_classifier_trainer.py",
    )
    violations = [
        _relative_repo_path(path) for path in forbidden_paths if path.exists()
    ]

    assert not violations, (
        "scripts federated_agent bridge는 payload-adapter 이름 파일을 소유하지 않는다. "
        "family별 payload/materialization 의미는 methods/adaptation/<family>/가 "
        "소유하고, scripts runtime adapter는 generic store/training bridge만 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_scripts_runtime_adapters_do_not_keep_federated_agent_monolith() -> None:
    monolith_path = SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_agent_runtime.py"
    package_root = SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_agent"
    expected_files = (
        package_root / "__init__.py",
        package_root / "artifact_store.py",
        package_root / "backend_resolver.py",
        package_root / "base_state_materialization.py",
        package_root / "client_update_flow.py",
        package_root / "peft_encoder_local_training.py",
        package_root / "row_validator.py",
        package_root / "scoring_runtime.py",
        package_root / "selection_runtime.py",
        package_root / "training_example_mapper.py",
        package_root / "training_runtime.py",
    )
    mapper_source = (package_root / "training_example_mapper.py").read_text(
        encoding="utf-8"
    )
    training_runtime_source = (package_root / "training_runtime.py").read_text(
        encoding="utf-8"
    )
    mapper_forbidden_snippets = (
        "WEAK_STRONG_PAIR_BACKEND_NAME",
        "RUNTIME_FALLBACK_TRAINING_PROFILE",
        "build_shared_adapter_training_backend",
        "LocalTrainingRequest(",
    )
    mapper_violations = [
        snippet for snippet in mapper_forbidden_snippets if snippet in mapper_source
    ]
    training_runtime_forbidden_snippets = (
        "methods.adaptation.peft_text_encoder",
        "LORA_CLASSIFIER_TRAINING_BACKEND_NAME",
        "PEFT_ENCODER_TRAINING_BACKEND_NAME",
        "PeftEncoderTrainingBackend",
        "simulation_inline_delta",
    )
    training_runtime_violations = [
        snippet
        for snippet in training_runtime_forbidden_snippets
        if snippet in training_runtime_source
    ]
    missing_files = [
        _relative_repo_path(path) for path in expected_files if not path.exists()
    ]
    generic_local_training_path = package_root / "local_training.py"

    assert not monolith_path.exists(), (
        "FL simulation agent runtime bridge는 federated_agent/ package의 책임별 "
        "module을 직접 import한다. 중앙 monolith/facade를 다시 만들지 않는다.\n"
        f"monolith path={_relative_repo_path(monolith_path)}"
    )
    assert not generic_local_training_path.exists(), (
        "PEFT encoder 전용 local training bridge는 peft_encoder_local_training.py에 "
        "둔다. generic local_training.py에 update-family별 분기를 누적하지 않는다."
    )
    assert not missing_files, (
        "federated_agent runtime adapter package는 artifact store, base-state "
        "materialization, local training, mapper, scoring/selection/training runtime "
        "bridge를 분리한다.\n"
        f"{chr(10).join(f'- {path}' for path in missing_files)}"
    )
    assert not mapper_violations, (
        "training_example_mapper는 row -> TrainingExampleSource 변환만 맡는다. "
        "backend fallback, weak/strong row 검증, local training request 생성은 "
        "각 전용 module로 분리한다.\n"
        f"violations={mapper_violations}"
    )
    assert not training_runtime_violations, (
        "training_runtime은 objective가 고른 backend를 registry로 resolve하고, "
        "backend가 제공하는 simulation capability만 호출한다. PEFT/Lora concrete "
        "이름과 inline executor wiring은 methods/adaptation/<family>/가 소유한다.\n"
        f"violations={training_runtime_violations}"
    )


def test_federated_agent_peft_round_files_do_not_own_update_submission() -> None:
    package_root = SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_agent"
    checked_paths = (
        package_root / "peft_encoder_method_owned_client_round.py",
        package_root / "peft_encoder_query_ssl_client_round.py",
    )
    required_helper = package_root / "client_update_flow.py"
    forbidden_snippets = (
        "ClientRoundSummary(",
        "client_update_submission",
        "SimulationClientArtifactStore(",
        "payload_byte_count(",
        "extract_delta_l2_norm(",
        "extract_aggregation_example_count(",
        "build_client_diagnostic_unlabeled_view",
    )
    violations: list[tuple[Path, str]] = []
    for path in checked_paths:
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in source:
                violations.append((_relative_repo_path(path), snippet))

    assert required_helper.exists(), (
        "family-specific client round 파일은 update submission/summary 조립을 "
        "반복 소유하지 않는다. 공통 흐름은 client_update_flow.py가 맡는다.\n"
        f"missing={_relative_repo_path(required_helper)}"
    )
    assert not violations, (
        "PEFT encoder client-round bridge는 local core 호출과 family upload 함수 "
        "선택만 맡는다. diagnostic view, server submit, payload byte, summary "
        "조립은 generic client update flow로 둔다.\n"
        f"{chr(10).join(f'- {path}: {snippet}' for path, snippet in violations)}"
    )


def test_peft_local_training_bridge_delegates_runtime_io_helpers() -> None:
    package_root = SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_agent"
    local_training_path = package_root / "peft_encoder_local_training.py"
    expected_owner_paths = (
        package_root / "base_state_materialization.py",
        package_root / "artifact_store.py",
    )
    source = local_training_path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "def _load_base_parameters_if_needed(",
        "def _load_base_partition_parameters_if_needed(",
        "def _save_agent_local_update(",
        "adapter_base_materialization_seconds",
        "adapter_base_partition_materialization_seconds",
        "agent_repository_save_seconds",
        "TrainingArtifactRepository(",
        "QuerySslLocalTrainingService(",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]
    missing_owner_paths = [
        _relative_repo_path(path) for path in expected_owner_paths if not path.exists()
    ]

    assert not missing_owner_paths, (
        "PEFT local training bridge의 runtime IO helper는 가장 가까운 capability "
        "module이 소유한다.\n"
        f"{chr(10).join(f'- {path}' for path in missing_owner_paths)}"
    )
    assert not violations, (
        "peft_encoder_local_training.py는 local training core 연결만 맡는다. "
        "base-state materialization timing은 base_state_materialization.py가, "
        "agent-local update 저장은 artifact_store.py가, agent service 생성은 "
        "training_runtime.py가 소유한다.\n"
        f"violations={violations}"
    )
