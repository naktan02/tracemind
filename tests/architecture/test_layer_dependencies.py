"""레이어 의존 규칙 아키텍처 테스트."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_SRC = REPO_ROOT / "shared" / "src"
METHODS_SRC = REPO_ROOT / "methods"
AGENT_SRC = REPO_ROOT / "agent" / "src"
AGENT_CONF = REPO_ROOT / "agent" / "conf"
MAIN_SERVER_SRC = REPO_ROOT / "main_server" / "src"
SCRIPTS_SRC = REPO_ROOT / "scripts"
SCRIPTS_RUNTIME_ADAPTER_SRC = SCRIPTS_SRC / "runtime_adapters"
EXPERIMENT_COMPILER_SRC = (
    MAIN_SERVER_SRC / "services" / "experiment_workspace" / "compiler"
)
FL_SIMULATION_IO_SRC = (
    SCRIPTS_SRC / "experiments" / "fl_ssl" / "federated_simulation" / "io"
)
QUERY_LORA_SSL_IO_SRC = SCRIPTS_SRC / "experiments" / "query_lora_ssl" / "io"
PYTHON_SOURCE_ROOTS = (
    SHARED_SRC,
    METHODS_SRC,
    AGENT_SRC,
    MAIN_SERVER_SRC,
    SCRIPTS_SRC,
    REPO_ROOT / "tests",
)
FORBIDDEN_DUNDER_ALL = "__" + "all__"
LEGACY_SHARED_PROTOTYPE_BUILDER_PATHS = (
    SHARED_SRC / "services" / "prototypes" / "build_strategies.py",
    SHARED_SRC / "services" / "prototypes" / "prototype_pack_builder.py",
)
PROTOTYPE_BUILDING_SRC = REPO_ROOT / "methods" / "prototype" / "building"
PROTOTYPE_SCORING_SRC = REPO_ROOT / "methods" / "prototype" / "scoring"
LEGACY_AGENT_QUERY_CLASSIFIER_ADAPTATION_SRC = (
    AGENT_SRC / "services" / "training" / "query_classifier_adaptation"
)

TEMPORARY_MAIN_SERVER_AGENT_IMPORT_EXCEPTIONS: set[Path] = set()
RUNTIME_LAYER_METHOD_NAME_FRAGMENTS = (
    "fedavg_pseudo_label",
    "fedmatch",
    "fedlgmatch",
    "fl2",
    "fixmatch",
    "freematch",
    "flexmatch",
    "comatch",
    "mixtext",
    "rdrop",
)


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.py") if "__pycache__" not in path.parts
    )


def _collect_absolute_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module)
    return imports


def _relative_repo_path(path: Path) -> Path:
    return path.relative_to(REPO_ROOT)


def _find_forbidden_imports(
    *,
    root: Path,
    forbidden_prefixes: tuple[str, ...],
    ignored_roots: tuple[Path, ...] = (),
) -> list[tuple[Path, str]]:
    violations: list[tuple[Path, str]] = []
    for path in _iter_python_files(root):
        if any(path.is_relative_to(ignored_root) for ignored_root in ignored_roots):
            continue
        imports = _collect_absolute_imports(path)
        for imported_module in sorted(imports):
            if imported_module.startswith(forbidden_prefixes):
                violations.append((_relative_repo_path(path), imported_module))
    return violations


def _format_violations(violations: list[tuple[Path, str]]) -> str:
    return "\n".join(
        f"- {path}: {imported_module}" for path, imported_module in violations
    )


def test_shared_layer_does_not_import_runtime_layers() -> None:
    violations = _find_forbidden_imports(
        root=SHARED_SRC,
        forbidden_prefixes=(
            "agent.src",
            "main_server.src",
            "methods",
            "research",
            "scripts",
        ),
    )
    assert not violations, _format_violations(violations)


def test_shared_contracts_do_not_keep_central_adapter_family_metadata_catalog() -> None:
    forbidden_path = SHARED_SRC / "contracts" / "adapter_family_metadata.py"
    assert not forbidden_path.exists(), (
        "shared는 중앙 adapter family metadata catalog를 소유하지 않는다. "
        "payload shape, adapter_kind, parse/serialize 규칙은 "
        "adapter_contract_families/<family>.py와 registry.py에 둔다."
    )


def test_shared_training_contracts_do_not_own_runtime_backend_defaults() -> None:
    path = SHARED_SRC / "contracts" / "training_contracts.py"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "DEFAULT_TRAINING_BACKEND_NAME",
        "diagonal_scale_heuristic",
        "lora_classifier_trainer",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "shared training contract는 runtime/backend 기본 선택값을 소유하지 않는다. "
        "training_backend_name은 payload 필수 값이고 기본 조합은 conf/ 또는 runtime "
        "default facade가 소유한다.\n"
        f"violations={violations}"
    )


def test_python_modules_do_not_define_dunder_all() -> None:
    violations = [
        _relative_repo_path(path)
        for root in PYTHON_SOURCE_ROOTS
        for path in _iter_python_files(root)
        if FORBIDDEN_DUNDER_ALL in path.read_text(encoding="utf-8")
    ]
    assert not violations, (
        "package-level export list는 사용하지 않는다. "
        "direct-file import로 공개 표면을 드러낸다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_init_modules_stay_marker_only() -> None:
    violations: list[Path] = []
    for root in PYTHON_SOURCE_ROOTS:
        for path in _iter_python_files(root):
            if path.name != "__init__.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            non_marker_nodes = [
                node
                for node in tree.body
                if not (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                )
            ]
            if non_marker_nodes:
                violations.append(_relative_repo_path(path))
    assert not violations, (
        "__init__.py는 package marker/docstring만 둔다. "
        "공개 표면은 direct-file import로 드러낸다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_prototype_builder_core_stays_in_methods_layer() -> None:
    existing_paths = [
        _relative_repo_path(path)
        for path in LEGACY_SHARED_PROTOTYPE_BUILDER_PATHS
        if path.exists()
    ]
    assert not existing_paths, (
        "prototype builder 알고리즘 core는 methods/prototype/building에 둔다. "
        f"legacy shared paths={sorted(str(path) for path in existing_paths)}"
    )


def test_prototype_building_keeps_strategy_files_separate() -> None:
    monolith_path = PROTOTYPE_BUILDING_SRC / "build_strategies.py"
    assert not monolith_path.exists(), (
        "prototype builder strategy는 base/single/kmeans/dbscan 파일로 나눈다. "
        f"monolith path={_relative_repo_path(monolith_path)}"
    )


def test_prototype_scoring_does_not_keep_policy_facade() -> None:
    facade_path = PROTOTYPE_SCORING_SRC / "policies.py"
    implementation_root = PROTOTYPE_SCORING_SRC / "score_policies"

    assert not facade_path.exists(), (
        "prototype score policy는 중앙 facade 없이 registry와 구현 파일로 분리한다. "
        "runtime은 policy_registry.py를, concrete 구현은 "
        "score_policies/<policy>.py를 직접 import한다.\n"
        f"facade path={_relative_repo_path(facade_path)}"
    )
    assert implementation_root.is_dir()


def test_methods_layer_does_not_import_runtime_or_research_layers() -> None:
    violations = _find_forbidden_imports(
        root=METHODS_SRC,
        forbidden_prefixes=("agent.src", "main_server.src", "research", "scripts"),
    )
    assert not violations, _format_violations(violations)


def test_query_classifier_adaptation_core_stays_in_methods_layer() -> None:
    existing_paths = [
        _relative_repo_path(path)
        for path in _iter_python_files(LEGACY_AGENT_QUERY_CLASSIFIER_ADAPTATION_SRC)
    ]
    assert not existing_paths, (
        "query classifier adaptation 학습 scaffold는 "
        "methods/adaptation/query_classifier_adaptation에 둔다. "
        "agent는 local runtime/API와 private state만 소유한다. "
        f"legacy paths={sorted(str(path) for path in existing_paths)}"
    )


def test_fl_local_update_profiles_do_not_keep_python_mapping_catalog() -> None:
    forbidden_path = METHODS_SRC / "federated_ssl" / "training_algorithm_profiles.py"
    assert not forbidden_path.exists(), (
        "FL local update profile 실행값은 conf/strategy_axes/fl/local_update_profile "
        "Hydra YAML이 소유한다. Python에는 profile별 objective mapping catalog를 "
        "다시 만들지 않는다."
    )


def test_runtime_layers_import_named_runtime_fallbacks_not_legacy_defaults() -> None:
    forbidden_modules = (
        "methods.federated_ssl.training_defaults",
        "methods.federated_ssl.training_default_values",
    )
    legacy_paths = (
        METHODS_SRC / "federated_ssl" / "training_defaults.py",
        METHODS_SRC / "federated_ssl" / "training_default_values.py",
    )
    existing_legacy_paths = [
        _relative_repo_path(path) for path in legacy_paths if path.exists()
    ]
    assert not existing_legacy_paths, (
        "짧은 compatibility facade가 내부 구조를 흐리지 않도록 legacy default "
        "module은 제거한다. runtime/API fallback은 runtime_fallbacks.py만 소유한다.\n"
        f"{chr(10).join(f'- {path}' for path in existing_legacy_paths)}"
    )

    violations: list[tuple[Path, str]] = []
    for root in (AGENT_SRC, MAIN_SERVER_SRC, SCRIPTS_SRC):
        violations.extend(
            _find_forbidden_imports(
                root=root,
                forbidden_prefixes=forbidden_modules,
            )
        )

    assert not violations, (
        "runtime 계층은 legacy training default module이 아니라 "
        "methods.federated_ssl.runtime_fallbacks를 import해야 한다. "
        "Hydra profile source-of-truth와 runtime fallback을 이름으로 분리한다.\n"
        f"{_format_violations(violations)}"
    )


def test_agent_layer_does_not_import_main_server_or_scripts() -> None:
    violations = _find_forbidden_imports(
        root=AGENT_SRC,
        forbidden_prefixes=("main_server.src", "research", "scripts"),
    )
    assert not violations, _format_violations(violations)


def test_agent_does_not_keep_unused_hydra_conf_tree() -> None:
    assert not AGENT_CONF.exists(), (
        "agent runtime 설정은 agent/conf Hydra tree로 두지 않는다. "
        "실험 조합은 루트 conf/, production runtime은 agent/src/config 또는 "
        "typed service wiring에서 소유한다."
    )


def test_main_server_layer_does_not_import_scripts() -> None:
    violations = _find_forbidden_imports(
        root=MAIN_SERVER_SRC,
        forbidden_prefixes=("research", "scripts"),
    )
    assert not violations, _format_violations(violations)


def test_main_server_agent_imports_are_limited_to_documented_exceptions() -> None:
    violations = _find_forbidden_imports(
        root=MAIN_SERVER_SRC,
        forbidden_prefixes=("agent.src",),
    )
    actual_exception_paths = {path for path, _ in violations}
    assert actual_exception_paths == TEMPORARY_MAIN_SERVER_AGENT_IMPORT_EXCEPTIONS, (
        "main_server -> agent 직접 의존은 현재 없어야 한다.\n"
        f"actual={sorted(str(path) for path in actual_exception_paths)}\n"
        "expected="
        f"{sorted(str(path) for path in TEMPORARY_MAIN_SERVER_AGENT_IMPORT_EXCEPTIONS)}"
    )


def test_runtime_layers_do_not_define_method_specific_modules() -> None:
    violations: list[Path] = []
    for root in (AGENT_SRC, MAIN_SERVER_SRC):
        for path in _iter_python_files(root):
            relative_path = _relative_repo_path(path)
            normalized_path = str(relative_path).lower()
            if any(
                method_fragment in normalized_path
                for method_fragment in RUNTIME_LAYER_METHOD_NAME_FRAGMENTS
            ):
                violations.append(relative_path)

    assert not violations, (
        "agent/main_server는 method-specific module을 소유하지 않는다. "
        "새 method 의미는 methods/에 두고 runtime 계층은 capability 이름의 "
        "port/adapter만 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_local_training_service_uses_update_executor_not_concrete_backends() -> None:
    path = (
        AGENT_SRC / "services" / "training" / "execution" / "local_training_service.py"
    )
    imports = _collect_absolute_imports(path)
    violations = sorted(
        imported_module
        for imported_module in imports
        if imported_module.startswith("agent.src.services.training.backends.training.")
    )
    assert not violations, (
        "LocalTrainingService는 selection orchestration만 맡고 update 생성은 "
        "LocalUpdateExecutor port를 통해 호출한다. concrete training backend나 "
        "training backend registry를 직접 import하지 않는다.\n"
        f"{chr(10).join(f'- {module}' for module in violations)}"
    )


def test_agent_training_backend_package_is_registry_facade_only() -> None:
    package_root = AGENT_SRC / "services" / "training" / "backends" / "training"
    allowed_files = {
        package_root / "__init__.py",
        package_root / "registry.py",
    }
    violations = [
        _relative_repo_path(path)
        for path in _iter_python_files(package_root)
        if path not in allowed_files
    ]

    assert not violations, (
        "agent training backend package는 methods-owned local update backend registry "
        "facade만 둔다. concrete local update backend는 "
        "methods/adaptation/<family>/training_backend.py에 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_fl_simulation_io_does_not_keep_artifact_facade() -> None:
    facade_path = FL_SIMULATION_IO_SRC / "artifacts.py"

    assert not facade_path.exists(), (
        "FL simulation artifact I/O는 중앙 artifacts.py facade 없이 writer/builder를 "
        "직접 호출한다. facade가 필요해 보이면 builder/writer 책임이 얕은지 먼저 "
        "점검한다.\n"
        f"facade path={_relative_repo_path(facade_path)}"
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


def test_query_lora_run_artifacts_do_not_keep_writer_exporter_monolith() -> None:
    orchestrator_path = QUERY_LORA_SSL_IO_SRC / "artifacts.py"
    expected_responsibility_files = (
        QUERY_LORA_SSL_IO_SRC / "artifact_paths.py",
        QUERY_LORA_SSL_IO_SRC / "artifact_writer.py",
        QUERY_LORA_SSL_IO_SRC / "manifest_builder.py",
        QUERY_LORA_SSL_IO_SRC / "model_artifact_exporter.py",
    )
    source = orchestrator_path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "json.dumps(",
        "torch.save(",
        ".write_text(",
        "save_pretrained(",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]
    missing_files = [
        _relative_repo_path(path)
        for path in expected_responsibility_files
        if not path.exists()
    ]

    assert not missing_files, (
        "Query LoRA run artifact 저장은 경로, 모델 export, payload build, JSON write "
        "책임 파일로 나눈다.\n"
        f"{chr(10).join(f'- {path}' for path in missing_files)}"
    )
    assert not violations, (
        "artifacts.py는 public orchestration entrypoint만 유지한다. 파일 저장, "
        "JSON serialization, model export를 다시 한 함수에 모으지 않는다.\n"
        f"violations={violations}"
    )


def test_query_lora_teacher_pseudo_label_does_not_keep_exporter_monolith() -> None:
    legacy_exporter_path = QUERY_LORA_SSL_IO_SRC / "teacher_pseudo_label_exporter.py"
    builder_path = QUERY_LORA_SSL_IO_SRC / "teacher_pseudo_label_builder.py"
    writer_path = QUERY_LORA_SSL_IO_SRC / "teacher_pseudo_label_artifact_writer.py"
    builder_source = builder_path.read_text(encoding="utf-8")
    builder_forbidden_snippets = (
        "json.dumps(",
        ".write_text(",
        ".open(",
        ".mkdir(",
    )
    violations = [
        snippet for snippet in builder_forbidden_snippets if snippet in builder_source
    ]

    assert not legacy_exporter_path.exists(), (
        "teacher pseudo-label 경로는 builder/writer를 직접 조합한다. "
        "단순 compatibility exporter facade를 다시 만들지 않는다.\n"
        f"legacy path={_relative_repo_path(legacy_exporter_path)}"
    )
    assert writer_path.exists(), (
        "teacher pseudo-label artifact 저장은 전용 writer가 맡는다. "
        f"missing writer={_relative_repo_path(writer_path)}"
    )
    assert not violations, (
        "TeacherPseudoLabelBuilder는 pseudo-label row와 diagnostics payload만 만든다. "
        "JSON serialization과 파일 write는 TeacherPseudoLabelArtifactWriter가 맡는다.\n"
        f"violations={violations}"
    )


def test_scripts_runtime_adapters_do_not_keep_federated_server_facade() -> None:
    facade_path = SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_server_runtime.py"

    assert not facade_path.exists(), (
        "FL simulation server runtime adapter는 federated_server/ package의 책임별 "
        "module을 직접 import한다. 중앙 re-export facade를 다시 만들지 않는다.\n"
        f"facade path={_relative_repo_path(facade_path)}"
    )


def test_scripts_runtime_adapters_do_not_keep_federated_agent_monolith() -> None:
    monolith_path = SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_agent_runtime.py"
    package_root = SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_agent"
    expected_files = (
        package_root / "__init__.py",
        package_root / "backend_resolver.py",
        package_root / "row_validator.py",
        package_root / "scoring_runtime.py",
        package_root / "selection_runtime.py",
        package_root / "training_example_mapper.py",
        package_root / "training_runtime.py",
    )
    mapper_source = (package_root / "training_example_mapper.py").read_text(
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
    missing_files = [
        _relative_repo_path(path) for path in expected_files if not path.exists()
    ]

    assert not monolith_path.exists(), (
        "FL simulation agent runtime bridge는 federated_agent/ package의 책임별 "
        "module을 직접 import한다. 중앙 monolith/facade를 다시 만들지 않는다.\n"
        f"monolith path={_relative_repo_path(monolith_path)}"
    )
    assert not missing_files, (
        "federated_agent runtime adapter package는 backend resolver, row validator, "
        "mapper, scoring/selection/training runtime bridge를 분리한다.\n"
        f"{chr(10).join(f'- {path}' for path in missing_files)}"
    )
    assert not mapper_violations, (
        "training_example_mapper는 row -> TrainingExampleSource 변환만 맡는다. "
        "backend fallback, weak/strong row 검증, local training request 생성은 "
        "각 전용 module로 분리한다.\n"
        f"violations={mapper_violations}"
    )


def test_scripts_reporting_does_not_wrap_shared_classification_report() -> None:
    facade_path = SCRIPTS_SRC / "reporting" / "classification_report.py"

    assert not facade_path.exists(), (
        "classification report canonical utility는 shared domain service가 소유한다. "
        "scripts/reporting에는 단순 re-export wrapper를 두지 않는다.\n"
        f"facade path={_relative_repo_path(facade_path)}"
    )


def test_scripts_do_not_wrap_shared_labeled_query_rows() -> None:
    facade_path = SCRIPTS_SRC / "io" / "labeled_query_rows.py"

    assert not facade_path.exists(), (
        "labeled query row canonical contract는 shared contract가 소유한다. "
        "scripts/io에는 단순 re-export wrapper를 두지 않는다.\n"
        f"facade path={_relative_repo_path(facade_path)}"
    )


def test_experiment_compiler_does_not_keep_policy_monolith() -> None:
    monolith_path = EXPERIMENT_COMPILER_SRC / "policies.py"

    assert not monolith_path.exists(), (
        "experiment compiler policy contract, registry primitive, default wiring, "
        "concrete policy 구현은 분리한다. 중앙 policies.py monolith를 다시 만들지 "
        "않는다.\n"
        f"monolith path={_relative_repo_path(monolith_path)}"
    )


def test_experiment_compiler_service_does_not_own_selection_or_override_building() -> (
    None
):
    service_path = EXPERIMENT_COMPILER_SRC / "service.py"
    required_files = (
        EXPERIMENT_COMPILER_SRC / "catalog_lookup.py",
        EXPERIMENT_COMPILER_SRC / "hydra_overrides.py",
        EXPERIMENT_COMPILER_SRC / "selection_compiler.py",
    )
    source = service_path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "for selection in manifest.selections",
        "selection.override_patch",
        "item.default_override_patch",
        "_format_hydra_value",
        "_merge_group_assignments",
        "_parse_hydra_override_map",
        "_selector_group_for_item",
        "_validate_selection_against_item",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]
    missing_files = [
        _relative_repo_path(path) for path in required_files if not path.exists()
    ]

    assert not missing_files, (
        "experiment compiler는 catalog lookup, selection compile, Hydra override "
        "utility를 service.py 밖의 전용 module로 분리한다.\n"
        f"{chr(10).join(f'- {path}' for path in missing_files)}"
    )
    assert not violations, (
        "ExperimentCompilerService는 compile orchestration만 맡는다. selection "
        "검증/selector build/override formatting 세부사항은 selection_compiler.py와 "
        "hydra_overrides.py가 맡는다.\n"
        f"violations={violations}"
    )


def test_main_server_round_family_package_has_no_concrete_family_modules() -> None:
    package_root = MAIN_SERVER_SRC / "services" / "federation" / "rounds" / "families"
    allowed_files = {
        package_root / "__init__.py",
        package_root / "models.py",
        package_root / "registry.py",
    }
    violations = [
        _relative_repo_path(path)
        for path in _iter_python_files(package_root)
        if path not in allowed_files
    ]

    assert not violations, (
        "main_server round family package는 shared adapter payload registry와 "
        "aggregation backend를 generic runtime으로 조합한다. concrete family "
        "module은 추가하지 않는다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_main_server_aggregation_package_is_executor_boundary_only() -> None:
    package_root = (
        MAIN_SERVER_SRC / "services" / "federation" / "rounds" / "aggregation"
    )
    allowed_files = {
        package_root / "__init__.py",
        package_root / "artifact_refs.py",
        package_root / "executor.py",
        package_root / "models.py",
        package_root / "registry.py",
    }
    violations = [
        _relative_repo_path(path)
        for path in _iter_python_files(package_root)
        if path not in allowed_files
    ]

    assert not violations, (
        "main_server aggregation package는 executor, registry, server-owned "
        "artifact ref capability만 둔다. FedAvg/FedProx 같은 aggregation method와 "
        "adapter-family projection은 methods/federated/aggregation이 소유한다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_main_server_aggregation_package_has_no_method_or_family_literals() -> None:
    package_root = (
        MAIN_SERVER_SRC / "services" / "federation" / "rounds" / "aggregation"
    )
    forbidden_snippets = (
        "fedavg",
        "fedprox",
        "diagonal_scale",
        "classifier_head",
        "lora_classifier",
        "FedAvg",
        "FedProx",
        "DiagonalScale",
        "ClassifierHead",
        "LoraClassifier",
    )
    violations: list[tuple[Path, str]] = []
    for path in _iter_python_files(package_root):
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in source:
                violations.append((_relative_repo_path(path), snippet))

    assert not violations, (
        "main_server aggregation package는 selected methods strategy를 실행하는 "
        "generic boundary만 둔다. aggregation method나 adapter family 상세 문자열은 "
        "methods/ 쪽 strategy/projection에 둔다.\n"
        f"{chr(10).join(f'- {path}: {snippet}' for path, snippet in violations)}"
    )


def test_main_server_aggregation_methods_do_not_define_family_specific_services() -> (
    None
):
    package_root = (
        MAIN_SERVER_SRC / "services" / "federation" / "rounds" / "aggregation"
    )
    from shared.src.contracts.adapter_contracts import AdapterKind

    family_name_prefixes = {
        "".join(part.capitalize() for part in adapter_kind.value.split("_"))
        for adapter_kind in AdapterKind
    }
    violations: list[tuple[Path, str]] = []
    for path in _iter_python_files(package_root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if any(
                node.name.startswith(prefix) for prefix in family_name_prefixes
            ) and (
                node.name.endswith("AggregationService")
                or node.name.endswith("AggregationConfig")
            ):
                violations.append((_relative_repo_path(path), node.name))

    assert not violations, (
        "main_server aggregation method file은 family별 service/config class를 "
        "누적하지 않는다. family 차이는 shared payload contract와 generic "
        "runtime spec 뒤에 둔다.\n"
        f"{chr(10).join(f'- {path}: {class_name}' for path, class_name in violations)}"
    )


def test_fedavg_strategy_file_stays_generic_without_family_specs() -> None:
    path = METHODS_SRC / "federated" / "aggregation" / "fedavg" / "strategy.py"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "DIAGONAL_SCALE",
        "CLASSIFIER_HEAD",
        "LORA_CLASSIFIER",
        "_FAMILY_METADATA",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "FedAvg strategy wiring 파일은 family별 projection/spec을 소유하지 않는다. "
        "family 상세는 methods/adaptation/<family>/fedavg.py와 "
        "fedavg_projection.py에 둔다.\n"
        f"violations={violations}"
    )


def test_fedavg_aggregation_package_stays_generic() -> None:
    package_root = METHODS_SRC / "federated" / "aggregation" / "fedavg"
    allowed_files = {
        package_root / "__init__.py",
        package_root / "strategy.py",
        package_root / "weighted_average.py",
    }
    violations = [
        _relative_repo_path(path)
        for path in _iter_python_files(package_root)
        if path not in allowed_files
    ]

    assert not violations, (
        "methods/federated/aggregation/fedavg는 FedAvg 공통 산술과 strategy wiring만 "
        "소유한다. adapter family별 FedAvg core와 payload projection은 "
        "methods/adaptation/<family>/에 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_main_server_federation_assets_package_has_no_source_modules() -> None:
    package_root = MAIN_SERVER_SRC / "services" / "federation" / "assets"
    violations = [
        _relative_repo_path(path)
        for path in _iter_python_files(package_root)
        if "__pycache__" not in path.parts
    ]

    assert not violations, (
        "main_server federation assets package는 넓은 catch-all source package로 "
        "사용하지 않는다. server-owned prototype artifact lifecycle은 "
        "main_server/src/services/federation/prototypes에 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_round_services_do_not_interpret_server_refs_as_paths() -> None:
    """server-owned ref 해석은 repository 계층에만 둔다."""

    forbidden_snippets = (
        "Path(update.payload_ref)",
        "Path(base_manifest.artifact_ref)",
        "Path(request.base_manifest.artifact_ref)",
        "load_shared_adapter_update_payload(Path(",
    )
    service_root = MAIN_SERVER_SRC / "services" / "federation" / "rounds"
    violations: list[tuple[Path, str]] = []
    for path in _iter_python_files(service_root):
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in text:
                violations.append((_relative_repo_path(path), snippet))

    assert not violations, (
        "payload_ref/artifact_ref는 opaque server-owned ref로 다룬다. "
        "파일 경로 compatibility는 infrastructure repository 안에만 둔다.\n"
        f"{chr(10).join(f'- {path}: {snippet}' for path, snippet in violations)}"
    )


def test_scripts_runtime_imports_stay_behind_documented_bridges() -> None:
    violations = _find_forbidden_imports(
        root=SCRIPTS_SRC,
        forbidden_prefixes=("agent.src", "main_server.src"),
        ignored_roots=(SCRIPTS_RUNTIME_ADAPTER_SRC,),
    )
    assert not violations, (
        "scripts는 agent/main_server 내부를 직접 import하지 않는다. "
        "runtime bridge는 scripts/runtime_adapters에 둔다.\n"
        f"{_format_violations(violations)}"
    )
