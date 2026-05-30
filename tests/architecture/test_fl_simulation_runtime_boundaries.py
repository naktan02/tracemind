"""FL simulation runtime adapter 아키텍처 guard."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_SRC = REPO_ROOT / "scripts"
SCRIPTS_RUNTIME_ADAPTER_SRC = SCRIPTS_SRC / "runtime_adapters"
METHODS_FEDERATED_SRC = REPO_ROOT / "methods" / "federated"
METHODS_FEDERATED_SSL_SRC = REPO_ROOT / "methods" / "federated_ssl"
CONF_SRC = REPO_ROOT / "conf"
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
    callable_loader_path = SCRIPTS_SRC / "support" / "configured_callable.py"
    checked_paths = (
        SCRIPTS_SRC / "experiments" / "fl_ssl" / "support" / "layout.py",
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
        SCRIPTS_SRC
        / "support"
        / "query_ssl_peft"
        / "query_ssl"
        / "view_preparation.py",
        SCRIPTS_SRC
        / "runtime_adapters"
        / "federated_server"
        / "initial_state_factory.py",
    )
    forbidden_snippet = "from importlib import import_module"
    violations = [
        _relative_repo_path(path)
        for path in checked_paths
        if forbidden_snippet in path.read_text(encoding="utf-8")
    ]

    assert callable_loader_path.exists(), (
        "Config가 선언한 callable import/validation은 scripts 공통 helper가 소유한다. "
        f"missing={_relative_repo_path(callable_loader_path)}"
    )
    assert not violations, (
        "FL simulation adapter들은 config-declared callable을 실행만 하고, "
        "fully-qualified path 파싱과 import 규칙은 "
        "scripts/support/configured_callable.py에 모은다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_server_step_policy_leaf_does_not_own_update_family_executor() -> None:
    policy_path = (
        CONF_SRC
        / "strategy_axes"
        / "fl_topology"
        / "server_step"
        / "supervised_seed_step.yaml"
    )
    update_family_path = (
        CONF_SRC
        / "strategy_axes"
        / "model_architecture"
        / "update_family"
        / "peft_text_encoder.yaml"
    )
    policy_source = policy_path.read_text(encoding="utf-8")
    update_family_source = update_family_path.read_text(encoding="utf-8")

    assert "executor:" not in policy_source, (
        "server_step_policy leaf는 server-side step 여부만 고른다. family-specific "
        "runtime executor는 update_family leaf가 소유한다.\n"
        f"path={_relative_repo_path(policy_path)}"
    )
    assert "server_step_executors:" in update_family_source, (
        "update family leaf가 server step policy 이름을 runtime executor로 매핑해야 "
        "prototype/linear 등 다음 model_architecture/update_family를 추가할 때 "
        "server_step_policy를 수정하지 않는다.\n"
        f"path={_relative_repo_path(update_family_path)}"
    )


def test_fl_entrypoint_does_not_embed_update_family_objective_payload_scope() -> None:
    entrypoint_path = (
        CONF_SRC / "entrypoints" / "fl_ssl" / "run_federated_simulation.yaml"
    )
    update_family_path = (
        CONF_SRC
        / "strategy_axes"
        / "model_architecture"
        / "update_family"
        / "peft_text_encoder.yaml"
    )
    entrypoint_source = entrypoint_path.read_text(encoding="utf-8")
    update_family_source = update_family_path.read_text(encoding="utf-8")

    assert "training_objective_payload_scope: peft_classifier" in update_family_source
    assert "    peft_classifier:\n" not in entrypoint_source, (
        "FL entrypoint는 PEFT text encoder objective extra scope를 직접 하드코딩하지 "
        "않는다. update_family leaf가 runtime payload를 어떤 objective scope로 "
        "주입할지 선언하고, script runner는 generic merge만 수행한다.\n"
        f"path={_relative_repo_path(entrypoint_path)}"
    )


def test_fl_aggregation_weight_policy_meaning_stays_in_methods() -> None:
    script_path = FL_SIMULATION_IO_SRC / "aggregation_diagnostics.py"
    methods_path = METHODS_FEDERATED_SRC / "aggregation_weighting.py"
    script_source = script_path.read_text(encoding="utf-8")
    methods_source = methods_path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "AGGREGATION_WEIGHT_UNIFORM",
        "AGGREGATION_WEIGHT_ACCEPTED_COUNT",
        "AGGREGATION_WEIGHT_EXAMPLE_COUNT",
        'policy.name == "uniform"',
        'policy.name == "accepted_count"',
        'policy.name == "example_count"',
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in script_source]

    assert "def aggregation_weight_for_diagnostics(" in methods_source
    assert "def aggregation_weight_basis_label(" in methods_source
    assert not violations, (
        "FL simulation report IO는 aggregation weight policy 의미를 직접 분기하지 "
        "않는다. policy별 weight/basis 해석은 methods/federated가 소유하고, "
        "scripts는 diagnostics payload 조립만 맡는다.\n"
        f"violations={violations}"
    )


def test_fl_simulation_diagnostic_sampling_core_stays_in_methods() -> None:
    checked_paths = (
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "adapters"
        / "diagnostic_view.py",
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "adapters"
        / "peer_probe.py",
    )
    forbidden_snippets = (
        "import random",
        "import hashlib",
        "random.Random(",
        "hashlib.sha256(",
        "defaultdict(",
    )
    violations = [
        (_relative_repo_path(path), snippet)
        for path in checked_paths
        for snippet in forbidden_snippets
        if snippet in path.read_text(encoding="utf-8")
    ]

    assert (METHODS_FEDERATED_SSL_SRC / "diagnostics" / "sampling.py").exists()
    assert not violations, (
        "FL diagnostic/probe row sampling algorithm은 methods/federated_ssl이 "
        "소유한다. scripts adapter는 config와 manifest 조립만 맡는다.\n"
        f"violations={violations}"
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
    generic_server_bridge_path = (
        SCRIPTS_RUNTIME_ADAPTER_SRC
        / "federated_server"
        / "generic_server_runtime_bridge.py"
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
        "build_peft_text_encoder_with_linear_head_from_config",
        "load_peft_encoder_base_parameters_into_model",
        "materialize_base_peft_encoder_state",
        "build_dataloader",
        "+ 7919",
        "sim_rev_{round_index:04d}_server_seed",
    )
    violations: list[tuple[Path, str]] = []
    source = generic_server_bridge_path.read_text(encoding="utf-8")
    for snippet in forbidden_snippets:
        if snippet in source:
            violations.append(
                (_relative_repo_path(generic_server_bridge_path), snippet)
            )
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


def test_fl_simulation_model_revision_policy_is_centralized() -> None:
    policy_path = (
        REPO_ROOT
        / "scripts"
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "model_revisions.py"
    )
    checked_paths = (
        REPO_ROOT
        / "scripts"
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "flow"
        / "bootstrap.py",
        REPO_ROOT
        / "scripts"
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "flow"
        / "round_loop.py",
        REPO_ROOT
        / "scripts"
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "io"
        / "communication_cost_estimates.py",
    )
    forbidden_snippets = ("sim_rev_0000", 'f"sim_rev_', "f'sim_rev_")
    violations: list[tuple[Path, str]] = []
    for path in checked_paths:
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in source:
                violations.append((_relative_repo_path(path), snippet))

    assert policy_path.exists(), (
        "FL simulation model revision naming은 한 helper에서 관리한다.\n"
        f"path={_relative_repo_path(policy_path)}"
    )
    assert not violations, (
        "bootstrap/round/report helper에 simulation revision 형식을 다시 박지 않는다. "
        "federated_simulation/model_revisions.py를 사용한다.\n"
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
        package_root / "generic_client_runtime_bridge.py",
        package_root / "scoring_runtime.py",
        package_root / "selection_runtime.py",
        package_root / "training_example_mapper.py",
        package_root / "training_runtime.py",
    )
    forbidden_paths = (package_root / "row_validator.py",)
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
    peft_encoder_local_training_path = package_root / "peft_encoder_local_training.py"

    assert not monolith_path.exists(), (
        "FL simulation agent runtime bridge는 federated_agent/ package의 책임별 "
        "module을 직접 import한다. 중앙 monolith/facade를 다시 만들지 않는다.\n"
        f"monolith path={_relative_repo_path(monolith_path)}"
    )
    assert not generic_local_training_path.exists(), (
        "PEFT encoder 전용 local training bridge는 peft_encoder_local_training.py에 "
        "둔다. generic local_training.py에 update-family별 분기를 누적하지 않는다."
    )
    assert not peft_encoder_local_training_path.exists(), (
        "PEFT encoder local training 파일은 dynamic loader/bridge 구조로 통합되어 "
        "더 이상 존재하지 않는다."
    )
    assert not any(path.exists() for path in forbidden_paths), (
        "training example backend별 row shape 요구사항은 methods/query_text_views가 "
        "소유하고, scripts runtime adapter는 별도 row_validator module을 두지 않는다."
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
    method_owned_round_path = package_root / "peft_encoder_method_owned_client_round.py"
    query_ssl_round_path = package_root / "peft_encoder_query_ssl_client_round.py"

    assert not method_owned_round_path.exists(), (
        "family-specific client round 파일은 generic bridge 구조 도입으로 "
        "더 이상 존재하지 않는다."
    )
    assert not query_ssl_round_path.exists(), (
        "family-specific client round 파일은 generic bridge 구조 도입으로 "
        "더 이상 존재하지 않는다."
    )


def test_peft_local_training_bridge_delegates_runtime_io_helpers() -> None:
    package_root = SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_agent"
    local_training_path = package_root / "peft_encoder_local_training.py"
    assert not local_training_path.exists(), (
        "PEFT encoder local training bridge 파일은 generic bridge로 통합되어 "
        "더 이상 존재하지 않는다."
    )
