"""레이어 의존 규칙 아키텍처 테스트."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

import yaml

from shared.src.contracts.adapter_contract_families.base import AdapterKind

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_SRC = REPO_ROOT / "conf"
SHARED_SRC = REPO_ROOT / "shared" / "src"
METHODS_SRC = REPO_ROOT / "methods"
CONF_FL_METHOD_DESCRIPTOR_SRC = (
    REPO_ROOT / "conf" / "strategy_axes" / "fl" / "method_descriptor"
)
CONF_FL_UPDATE_PARTITION_POLICY_SRC = (
    REPO_ROOT / "conf" / "strategy_axes" / "fl" / "update_partition_policy"
)
CONF_FL_PEER_CONTEXT_POLICY_SRC = (
    REPO_ROOT / "conf" / "strategy_axes" / "fl" / "peer_context_policy"
)
AGENT_SRC = REPO_ROOT / "agent" / "src"
AGENT_CONF = REPO_ROOT / "agent" / "conf"
MAIN_SERVER_SRC = REPO_ROOT / "main_server" / "src"
SCRIPTS_SRC = REPO_ROOT / "scripts"
SCRIPTS_RUNTIME_ADAPTER_SRC = SCRIPTS_SRC / "runtime_adapters"
FL_SIMULATION_IO_SRC = (
    SCRIPTS_SRC / "experiments" / "fl_ssl" / "federated_simulation" / "io"
)
QUERY_LORA_SSL_IO_SRC = SCRIPTS_SRC / "experiments" / "query_lora_ssl" / "io"
PROTOTYPE_STRATEGY_SRC = (
    SCRIPTS_SRC / "experiments" / "prototype_analysis" / "prototype_strategy"
)
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
METHODS_FEDERATED_SSL_SRC = METHODS_SRC / "federated_ssl"
PEFT_TEXT_CLASSIFIER_SRC = METHODS_SRC / "adaptation" / "peft_text_classifier"
PEFT_TEXT_CLASSIFIER_AGGREGATION_SRC = PEFT_TEXT_CLASSIFIER_SRC / "aggregation"
LINEAR_HEAD_CLASSIFICATION_SRC = METHODS_SRC / "classification" / "linear_head"
PEFT_ADAPTERS_SRC = METHODS_SRC / "adaptation" / "peft_adapters"
LEGACY_AGENT_QUERY_TEXT_VIEWS_SRC = (
    AGENT_SRC / "services" / "training" / "query_text_views"
)

TEMPORARY_MAIN_SERVER_AGENT_IMPORT_EXCEPTIONS: set[Path] = set()
RUNTIME_LAYER_METHOD_NAME_FRAGMENTS = (
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
FL_SCRIPT_RUNTIME_ROOTS = (
    SCRIPTS_SRC / "experiments" / "fl_ssl",
    SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_agent",
    SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_server",
)
PAPER_METHOD_NAME_FRAGMENTS = (
    "fedmatch",
    "fedlgmatch",
    "fl2",
    "fl_2",
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


def test_shared_adapter_contracts_do_not_keep_legacy_facade() -> None:
    forbidden_path = SHARED_SRC / "contracts" / "adapter_contracts.py"
    assert not forbidden_path.exists(), (
        "shared adapter payload contract는 adapter_contract_families/의 family별 "
        "module과 base/factories/io/registry를 direct import한다. "
        "legacy compatibility facade인 adapter_contracts.py는 재도입하지 않는다."
    )


def test_python_modules_do_not_import_legacy_shared_adapter_contract_facade() -> None:
    legacy_module = "shared.src.contracts.adapter_contracts"
    violations: list[tuple[Path, str]] = []
    for root in PYTHON_SOURCE_ROOTS:
        violations.extend(
            _find_forbidden_imports(
                root=root,
                forbidden_prefixes=(legacy_module,),
            )
        )
    assert not violations, (
        "shared adapter payload는 adapter_contract_families/ direct import를 사용한다. "
        "legacy facade import를 재도입하지 않는다.\n"
        f"{_format_violations(violations)}"
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


def test_shared_training_contracts_do_not_own_adapter_payload_format_catalog() -> None:
    path = SHARED_SRC / "contracts" / "training_contracts.py"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "class UpdatePayloadFormat",
        "DIAGONAL_SCALE_UPDATE",
        "CLASSIFIER_HEAD_UPDATE",
        "LORA_CLASSIFIER_UPDATE",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "shared training envelope는 payload_format 문자열 필드만 소유한다. "
        "adapter-family별 canonical/accepted format은 "
        "shared/src/contracts/adapter_contract_families/<family>.py가 소유한다.\n"
        f"violations={violations}"
    )


def test_shared_adapter_base_does_not_default_to_diagonal_scale() -> None:
    checked_paths = (
        SHARED_SRC / "contracts" / "adapter_contract_families" / "base.py",
        SHARED_SRC / "contracts" / "adapter_contract_families" / "registry.py",
    )
    forbidden_snippets = (
        "default=AdapterKind.DIAGONAL_SCALE.value",
        'data.get("adapter_kind", AdapterKind.DIAGONAL_SCALE.value)',
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for path in checked_paths
        for snippet in forbidden_snippets
        if snippet in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "shared adapter base/registry는 새 payload를 diagonal_scale로 암묵 해석하지 "
        "않는다. legacy vector_adapter schema compatibility만 명시적으로 허용한다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
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


def test_hydra_config_groups_are_python_package_markers() -> None:
    config_group_dirs = sorted(
        path.parent
        for path in CONF_SRC.rglob("*.yaml")
        if "__pycache__" not in path.parts
    )
    violations = [
        _relative_repo_path(path)
        for path in dict.fromkeys(config_group_dirs)
        if not (path / "__init__.py").exists()
    ]

    assert not violations, (
        "conf/** Hydra config group directory는 config_module='conf' import 경계를 "
        "명확히 하기 위해 __init__.py package marker를 둔다.\n"
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


def test_query_text_views_core_stays_in_methods_layer() -> None:
    existing_paths = [
        _relative_repo_path(path)
        for path in _iter_python_files(LEGACY_AGENT_QUERY_TEXT_VIEWS_SRC)
    ]
    assert not existing_paths, (
        "query text view/token-batch glue는 "
        "methods/adaptation/query_text_views에 둔다. "
        "agent는 local runtime/API와 private state만 소유한다. "
        f"legacy paths={sorted(str(path) for path in existing_paths)}"
    )


def test_legacy_query_classifier_adaptation_package_is_removed() -> None:
    legacy_root = METHODS_SRC / "adaptation" / "query_classifier_adaptation"
    assert not legacy_root.exists(), (
        "query input/view glue의 canonical package는 "
        "methods/adaptation/query_text_views다. legacy "
        "methods/adaptation/query_classifier_adaptation package를 다시 만들지 않는다."
    )


def test_query_text_views_stays_input_glue_only() -> None:
    violations = _find_forbidden_imports(
        root=METHODS_SRC / "adaptation" / "query_text_views",
        forbidden_prefixes=(
            "methods.adaptation.lora_classifier",
            "methods.adaptation.peft_text_classifier",
            "methods.adaptation.peft_adapters",
            "shared.src.contracts.adapter_contract_families",
            "shared.src.domain.entities.training.shared_adapter_state",
            "shared.src.domain.entities.training.shared_adapter_update",
        ),
    )

    assert not violations, (
        "query_text_views는 query-domain row/view/token-batch 입력 glue만 "
        "소유한다. PEFT model composition, shared update payload, adapter-family "
        "materialization은 각 canonical owner에 둔다.\n"
        f"{_format_violations(violations)}"
    )


def test_local_objective_regularizers_stay_update_payload_agnostic() -> None:
    violations = _find_forbidden_imports(
        root=METHODS_SRC / "adaptation" / "local_objective_regularizers",
        forbidden_prefixes=(
            "agent.src",
            "main_server.src",
            "scripts",
            "methods.federated_ssl.fedmatch",
            "shared.src.contracts.adapter_contract_families",
            "shared.src.domain.entities.training.shared_adapter_state",
            "shared.src.domain.entities.training.shared_adapter_update",
        ),
    )

    assert not violations, (
        "local_objective_regularizers는 local loss regularization만 소유한다. "
        "shared payload, server aggregation, method-specific round policy가 필요하면 "
        "별도 capability로 분리한다.\n"
        f"{_format_violations(violations)}"
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


def test_round_manager_does_not_own_default_adapter_family() -> None:
    path = (
        MAIN_SERVER_SRC
        / "services"
        / "federation"
        / "rounds"
        / "round_manager_service.py"
    )
    source = path.read_text(encoding="utf-8")
    imports = _collect_absolute_imports(path)

    assert "main_server.src.services.federation.rounds.families.registry" not in imports
    assert "build_shared_adapter_round_family" not in source
    assert "diagonal_scale" not in source, (
        "RoundManagerService는 round lifecycle orchestration만 소유한다. no-config "
        "legacy adapter family fallback은 runtime/config profile에 격리하고, "
        "service는 caller가 조립한 adapter_family를 받는다."
    )


def test_server_round_runtime_config_isolates_legacy_adapter_profile() -> None:
    path = (
        MAIN_SERVER_SRC / "services" / "federation" / "rounds" / "runtime" / "config.py"
    )
    imports = _collect_absolute_imports(path)

    assert (
        "shared.src.contracts.adapter_contract_families.diagonal_scale" not in imports
    ), (
        "server runtime config는 shared diagonal_scale contract를 직접 import하지 "
        "않는다. legacy no-config fallback은 named runtime profile 값으로만 "
        "격리한다."
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


def test_fl_scripts_do_not_define_paper_method_specific_runtime_modules() -> None:
    violations: list[Path] = []
    for root in FL_SCRIPT_RUNTIME_ROOTS:
        for path in _iter_python_files(root):
            relative_path = _relative_repo_path(path)
            normalized_path = str(relative_path).lower()
            if any(
                method_fragment in normalized_path
                for method_fragment in PAPER_METHOD_NAME_FRAGMENTS
            ):
                violations.append(relative_path)

    assert not violations, (
        "FL scripts/runtime adapters는 FedMatch/FedLGMatch/(FL)^2 같은 논문 method "
        "구현을 파일명으로 소유하지 않는다. method identity와 policy 의미는 "
        "methods/federated_ssl/<method>/에 두고, scripts는 entrypoint/report/runtime "
        "bridge만 맡긴다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_fl_scripts_legacy_family_names_stay_in_declared_compatibility_files() -> None:
    roots = (
        SCRIPTS_SRC / "experiments" / "fl_ssl" / "federated_simulation",
        SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_agent",
    )
    legacy_snippets = (
        "lora_classifier",
        "peft_classifier",
        "fedmatch_helper_count",
        "expect_lora_classifier_aggregate_snapshot",
    )
    allowed_paths = {
        Path("scripts/experiments/fl_ssl/federated_simulation/adapters/evaluation.py"),
        Path(
            "scripts/experiments/fl_ssl/federated_simulation/adapters/"
            "server_step_execution.py"
        ),
        Path("scripts/experiments/fl_ssl/federated_simulation/config_request.py"),
        Path("scripts/experiments/fl_ssl/federated_simulation/flow/bootstrap.py"),
        Path(
            "scripts/experiments/fl_ssl/federated_simulation/io/report_verification.py"
        ),
        Path(
            "scripts/experiments/fl_ssl/federated_simulation/io/"
            "report_verification_models.py"
        ),
        Path("scripts/experiments/fl_ssl/federated_simulation/models.py"),
        Path("scripts/runtime_adapters/federated_agent/base_state_materialization.py"),
        Path("scripts/runtime_adapters/federated_agent/local_training.py"),
    }
    actual_paths: set[Path] = set()
    for root in roots:
        for path in _iter_python_files(root):
            source = path.read_text(encoding="utf-8")
            if any(snippet in source for snippet in legacy_snippets):
                actual_paths.add(_relative_repo_path(path))

    assert actual_paths <= allowed_paths, (
        "FL scripts/runtime adapters에 adapter-family/method legacy 이름을 새 파일로 "
        "확산하지 않는다. 남은 lora_classifier/peft_classifier/FedMatch report "
        "문자열은 docs/contracts/legacy_contract_ledger.md에 기록한 compatibility "
        "표면으로만 허용한다.\n"
        f"{chr(10).join(f'- {path}' for path in sorted(actual_paths - allowed_paths))}"
    )


def test_production_federated_ssl_methods_do_not_keep_dummy_extensions() -> None:
    package_root = METHODS_SRC / "federated_ssl"
    forbidden_path_fragments = ("dummy", "test_only")
    violations = [
        _relative_repo_path(path)
        for path in _iter_python_files(package_root)
        if any(
            fragment in str(_relative_repo_path(path)).lower()
            for fragment in forbidden_path_fragments
        )
    ]

    assert not violations, (
        "Batch 7 extension dry run은 tests/fixtures 아래 test-only method로 검증한다. "
        "production methods/federated_ssl에는 dummy/test-only method 파일을 남기지 "
        "않는다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_fl_method_descriptor_configs_point_to_real_method_modules() -> None:
    """method descriptor YAML만 먼저 생기는 placeholder config를 막는다."""

    violations: list[str] = []
    method_package_root = METHODS_SRC / "federated_ssl"
    for config_path in sorted(CONF_FL_METHOD_DESCRIPTOR_SRC.glob("*.yaml")):
        method_name = config_path.stem
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        declared_name = payload.get("name")
        method_dir = method_package_root / method_name

        if declared_name != method_name:
            violations.append(
                f"{_relative_repo_path(config_path)}: name={declared_name!r} "
                f"must match filename stem {method_name!r}"
            )
        if payload.get("use_original_parameters") is True:
            duplicated_keys = sorted(
                key
                for key in ("original_parameters", "effective_parameters")
                if key in payload
            )
            if duplicated_keys:
                violations.append(
                    f"{_relative_repo_path(config_path)}: original method parameter "
                    "values must stay in methods/federated_ssl/<method>/"
                    f"original_spec.py, not YAML: {duplicated_keys}"
                )
            round_state_exchange = payload.get("round_state_exchange")
            if isinstance(round_state_exchange, dict):
                duplicated_round_state_keys = sorted(
                    key
                    for key in (
                        "num_helpers",
                        "refresh_interval",
                        "helper_refresh_interval",
                    )
                    if key in round_state_exchange
                )
                if duplicated_round_state_keys:
                    violations.append(
                        f"{_relative_repo_path(config_path)}: original helper "
                        "parameter values must stay in "
                        "methods/federated_ssl/<method>/original_spec.py, not "
                        "round_state_exchange YAML: "
                        f"{duplicated_round_state_keys}"
                    )
        if not method_dir.is_dir():
            violations.append(
                f"{_relative_repo_path(config_path)}: missing "
                f"{_relative_repo_path(method_dir)}"
            )
            continue

        required_files = (
            method_dir / "descriptor.py",
            method_dir / "local_objective.py",
            method_dir / "server_policy.py",
            method_dir / "round_policy.py",
        )
        for required_file in required_files:
            if not required_file.is_file():
                violations.append(
                    f"{_relative_repo_path(config_path)}: missing "
                    f"{_relative_repo_path(required_file)}"
                )
        registry_wiring_shim = method_dir / f"{method_name}.py"
        if registry_wiring_shim.exists():
            violations.append(
                f"{_relative_repo_path(config_path)}: remove pass-through "
                f"registry wiring shim {_relative_repo_path(registry_wiring_shim)}; "
                "descriptor.py의 descriptor 변수를 registry convention으로 등록한다"
            )

    assert not violations, (
        "FL method descriptor config는 실제 methods/federated_ssl/<method>/ 구현이 "
        "존재한 뒤에만 추가한다. 선택 전 placeholder YAML은 두지 않는다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_fl_update_partition_policy_configs_stay_mechanism_only() -> None:
    """공통 partition capability에 method-local scheme 이름을 올리지 않는다."""

    violations: list[str] = []
    method_local_fragments = ("sigma", "psi")
    for config_path in sorted(CONF_FL_UPDATE_PARTITION_POLICY_SRC.glob("*.yaml")):
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        policy_name = str(payload.get("name", "")).strip()
        checked_values = (config_path.stem, policy_name)
        if any(
            fragment in value.lower()
            for value in checked_values
            for fragment in method_local_fragments
        ):
            violations.append(
                f"{_relative_repo_path(config_path)}: method-local partition scheme "
                "names such as sigma/psi must stay in methods/federated_ssl/<method>/"
            )
        if policy_name == "partitioned":
            if "partitions" in payload:
                violations.append(
                    f"{_relative_repo_path(config_path)}: generic partitioned "
                    "capability must not declare method-local partition names"
                )
            if payload.get("partition_scheme_source") != "method_descriptor":
                violations.append(
                    f"{_relative_repo_path(config_path)}: partitioned capability "
                    "must point partition_scheme_source to method_descriptor"
                )

    assert not violations, (
        "FL update_partition_policy config는 unified/partitioned 같은 mechanism만 "
        "표현한다. sigma/psi 같은 scheme 이름과 routing 의미는 method package가 "
        "소유한다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_federated_ssl_capability_axes_do_not_split_tiny_policy_files() -> None:
    forbidden_paths = (
        METHODS_FEDERATED_SSL_SRC / "local_ssl_policy.py",
        METHODS_FEDERATED_SSL_SRC / "server_update_policy.py",
    )
    violations = [
        _relative_repo_path(path) for path in forbidden_paths if path.exists()
    ]

    assert not violations, (
        "FL SSL local/server capability 이름과 작은 normalizer는 "
        "capability_axes.py에 함께 둔다. 이름/상수만 가진 sibling policy 파일은 "
        "reader path를 늘린다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_federated_ssl_capability_axes_stays_adapter_family_agnostic() -> None:
    path = METHODS_FEDERATED_SSL_SRC / "capability_axes.py"
    imports = _collect_absolute_imports(path)
    forbidden_imports = {
        "shared.src.contracts.adapter_contract_families.classifier_head",
        "shared.src.contracts.adapter_contract_families.diagonal_scale",
        "shared.src.contracts.adapter_contract_families.lora_classifier",
    }
    source = path.read_text(encoding="utf-8")

    assert not sorted(imports & forbidden_imports), (
        "FL SSL capability axis는 local/server policy 이름만 소유한다. "
        "adapter-family payload contract나 runtime backend 해석은 "
        "methods/adaptation/<family>/federated_ssl/가 소유한다."
    )
    assert "lora_classifier" not in source, (
        "capability_axes.py는 LoRA-classifier family literal을 하드코딩하지 않는다."
    )


def test_fedmatch_descriptor_does_not_keep_recipe_pass_through() -> None:
    recipe_path = METHODS_FEDERATED_SSL_SRC / "fedmatch" / "recipe.py"

    assert not recipe_path.exists(), (
        "FedMatch recipe metadata는 descriptor.py에서 바로 읽는다. descriptor.recipe를 "
        "다시 노출하는 pass-through recipe.py는 만들지 않는다.\n"
        f"recipe path={_relative_repo_path(recipe_path)}"
    )


def test_federated_ssl_method_packages_do_not_own_adapter_family_runtime_files() -> (
    None
):
    forbidden_fragments = (
        "lora_classifier",
        "classifier_head",
        "diagonal_scale",
        "full_encoder",
        "dora",
    )
    violations: list[Path] = []
    for method_dir in sorted(METHODS_FEDERATED_SSL_SRC.iterdir()):
        if not method_dir.is_dir() or method_dir.name.startswith("__"):
            continue
        for path in _iter_python_files(method_dir):
            if any(fragment in path.stem.lower() for fragment in forbidden_fragments):
                violations.append(_relative_repo_path(path))

    assert not violations, (
        "methods/federated_ssl/<method>/는 논문 method 의미와 policy를 소유한다. "
        "LoRA-classifier/full encoder/DoRA 같은 adapter-family 실행 구현은 "
        "methods/adaptation/<family>/federated_ssl/에 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_adapter_family_federated_ssl_files_do_not_multiply_by_method_name() -> None:
    method_fragments = (
        "fedmatch",
        "fedlgmatch",
        "fl2",
        "fixmatch",
        "flexmatch",
        "freematch",
    )
    violations: list[Path] = []
    for family_dir in sorted((METHODS_SRC / "adaptation").iterdir()):
        package_root = family_dir / "federated_ssl"
        if not package_root.is_dir():
            continue
        for path in _iter_python_files(package_root):
            if any(fragment in path.stem.lower() for fragment in method_fragments):
                violations.append(_relative_repo_path(path))

    assert not violations, (
        "methods/adaptation/<family>/federated_ssl/는 adapter-family 실행 primitive를 "
        "소유한다. 새 FL SSL method마다 <method>_*.py 파일을 늘리지 말고 "
        "method 의미는 methods/federated_ssl/<method>/에 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_lora_classifier_partitioned_training_loop_is_method_neutral() -> None:
    path = (
        METHODS_SRC
        / "adaptation"
        / "peft_text_classifier"
        / "federated_ssl"
        / "partitioned"
        / "training_loop.py"
    )
    imports = _collect_absolute_imports(path)
    violations = sorted(
        imported
        for imported in imports
        if imported.startswith("methods.federated_ssl.fedmatch")
    )

    assert not violations, (
        "partitioned training loop는 adapter-family execution primitive다. "
        "FedMatch objective와 partition 이름은 methods/federated_ssl/fedmatch/의 "
        "caller가 주입해야 한다.\n"
        f"{chr(10).join(f'- {item}' for item in violations)}"
    )


def test_methods_lora_classifier_compatibility_package_is_removed() -> None:
    legacy_root = METHODS_SRC / "adaptation" / "lora_classifier"
    existing_paths = _existing_non_cache_paths((legacy_root,))

    assert not existing_paths, (
        "methods/adaptation/lora_classifier는 더 이상 internal compatibility "
        "package로 유지하지 않는다. 구현 source of truth는 "
        "methods/adaptation/peft_text_classifier/**이고, v1 lora_classifier 이름은 "
        "shared contract/artifact reader compatibility 표면에만 남긴다.\n"
        f"{chr(10).join(f'- {path}' for path in existing_paths)}"
    )


def test_internal_code_does_not_import_legacy_lora_classifier_methods_package() -> None:
    violations: list[tuple[Path, str]] = []
    for root in PYTHON_SOURCE_ROOTS:
        violations.extend(
            _find_forbidden_imports(
                root=root,
                forbidden_prefixes=("methods.adaptation.lora_classifier",),
            )
        )

    assert not violations, (
        "methods.adaptation.lora_classifier import 경로는 삭제된 compatibility "
        "package다. 새 internal code는 peft_text_classifier 경로를 직접 import한다.\n"
        f"{_format_violations(violations)}"
    )


def test_peft_text_classifier_does_not_import_fedmatch_method() -> None:
    violations = _find_forbidden_imports(
        root=PEFT_TEXT_CLASSIFIER_SRC,
        forbidden_prefixes=("methods.federated_ssl.fedmatch",),
    )

    assert not violations, (
        "methods/adaptation/peft_text_classifier/**는 PEFT text classifier 실행 "
        "primitive를 "
        "소유한다. FedMatch 의미, partition routing, original parameter는 "
        "methods/federated_ssl/fedmatch/에서 callable/config로 주입한다.\n"
        f"{_format_violations(violations)}"
    )


def test_peft_text_classifier_does_not_depend_on_legacy_lora_classifier() -> None:
    violations = _find_forbidden_imports(
        root=PEFT_TEXT_CLASSIFIER_SRC,
        forbidden_prefixes=(
            "methods.adaptation.classifier_head",
            "methods.adaptation.lora_classifier",
        ),
    )

    assert not violations, (
        "새 peft_text_classifier 내부 코드는 legacy classifier_head/lora_classifier "
        "경로를 import하지 않는다. 기존 경로는 migration shim으로만 남기고, 내부 "
        "source of truth는 peft_text_classifier 아래에 둔다.\n"
        f"{_format_violations(violations)}"
    )


def test_classification_adaptation_is_modality_independent() -> None:
    violations = _find_forbidden_imports(
        root=LINEAR_HEAD_CLASSIFICATION_SRC,
        forbidden_prefixes=(
            "methods.adaptation.classifier_head",
            "methods.adaptation.lora_classifier",
            "methods.adaptation.peft_text_classifier",
            "methods.adaptation.text_classifier",
        ),
    )

    assert not violations, (
        "methods/classification/linear_head/**는 modality-independent classification "
        "primitive를 소유한다. text-specific PEFT encoder나 legacy classifier_head "
        "경로를 import하지 않는다.\n"
        f"{_format_violations(violations)}"
    )


def test_peft_text_classifier_uses_peft_adapters_axis() -> None:
    violations = _find_forbidden_imports(
        root=PEFT_TEXT_CLASSIFIER_SRC,
        forbidden_prefixes=(
            "methods.adaptation.lora.",
            "methods.adaptation.peft.",
        ),
    )

    assert not violations, (
        "PEFT encoder text classifier는 LoRA/DoRA mechanism을 "
        "methods/adaptation/peft_adapters/** 축으로만 참조한다. legacy "
        "methods/adaptation/lora 또는 methods/adaptation/peft 경로에 묶지 않는다.\n"
        f"{_format_violations(violations)}"
    )


def test_legacy_peft_adapter_packages_are_removed() -> None:
    legacy_paths = (
        METHODS_SRC / "adaptation" / "peft",
        METHODS_SRC / "adaptation" / "lora",
    )
    existing_paths = _existing_non_cache_paths(legacy_paths)

    assert not existing_paths, (
        "PEFT mechanism source of truth는 methods/adaptation/peft_adapters/**다. "
        "legacy methods/adaptation/peft, methods/adaptation/lora package는 "
        "compatibility phase 종료 후 다시 만들지 않는다.\n"
        f"{chr(10).join(f'- {path}' for path in existing_paths)}"
    )


def test_legacy_classifier_head_packages_are_removed() -> None:
    legacy_paths = (
        METHODS_SRC / "adaptation" / "classifier_head",
        METHODS_SRC / "adaptation" / "classification",
        METHODS_SRC / "adaptation" / "text_classifier",
    )
    existing_paths = _existing_non_cache_paths(legacy_paths)

    assert not existing_paths, (
        "feature-head classification source of truth는 "
        "methods/classification/linear_head/**다. legacy classifier_head와 "
        "text_classifier shim package는 compatibility phase 종료 후 "
        "다시 만들지 않는다.\n"
        f"{chr(10).join(f'- {path}' for path in existing_paths)}"
    )


def _existing_non_cache_paths(paths: Sequence[Path]) -> list[Path]:
    existing_paths: list[Path] = []
    for path in paths:
        if path.is_file():
            existing_paths.append(_relative_repo_path(path))
        elif path.is_dir():
            for child in _iter_python_files(path):
                existing_paths.append(_relative_repo_path(child))
            existing_paths.extend(
                _relative_repo_path(child)
                for child in path.rglob("*.md")
                if "__pycache__" not in child.parts
            )
    return sorted(existing_paths)


def test_adaptation_aggregation_files_stay_projection_only() -> None:
    violations: list[Path] = []
    aggregation_roots = (
        PEFT_TEXT_CLASSIFIER_AGGREGATION_SRC,
        LINEAR_HEAD_CLASSIFICATION_SRC / "aggregation",
    )
    for aggregation_root in aggregation_roots:
        if not aggregation_root.is_dir():
            continue
        for path in _iter_python_files(aggregation_root):
            source = path.read_text(encoding="utf-8")
            if path.stem.endswith("_projection"):
                continue
            if "weighted_average" in source or "def fedavg" in source.lower():
                violations.append(_relative_repo_path(path))

    assert not violations, (
        "classification/peft_text_classifier aggregation 계층은 family state를 generic "
        "aggregation input/output으로 바꾸는 projection만 소유한다. "
        "weighted average policy와 FedAvg algorithm은 methods/federated/aggregation/에 "
        "둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_peft_adapters_do_not_import_classifier_task_payloads() -> None:
    violations = _find_forbidden_imports(
        root=PEFT_ADAPTERS_SRC,
        forbidden_prefixes=(
            "methods.adaptation.classifier_head",
            "methods.adaptation.lora_classifier",
            "methods.adaptation.peft_text_classifier",
            "methods.adaptation.text_classifier",
            "shared.src.contracts.adapter_contract_families.classifier_head",
            "shared.src.contracts.adapter_contract_families.lora_classifier",
        ),
    )

    assert not violations, (
        "methods/adaptation/peft_adapters/**는 LoRA/DoRA 같은 PEFT mechanism만 "
        "소유한다. classifier label, task head, update payload 의미는 "
        "peft_text_classifier adaptation 또는 shared contract가 소유한다.\n"
        f"{_format_violations(violations)}"
    )


def test_fl_peer_context_policy_configs_stay_mechanism_only() -> None:
    """공통 peer context capability에 FedMatch helper 기본값을 올리지 않는다."""

    violations: list[str] = []
    method_parameter_keys = {"num_helpers", "refresh_interval", "h_interval"}
    for config_path in sorted(CONF_FL_PEER_CONTEXT_POLICY_SRC.glob("*.yaml")):
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        policy_name = str(payload.get("name", "")).strip()
        leaked_keys = sorted(method_parameter_keys.intersection(payload))
        if leaked_keys:
            violations.append(
                f"{_relative_repo_path(config_path)}: method-local peer context "
                f"parameters must stay in method descriptors: {leaked_keys}"
            )
        if policy_name != "none" and payload.get("parameter_source") != (
            "method_descriptor"
        ):
            violations.append(
                f"{_relative_repo_path(config_path)}: non-empty peer context "
                "capability must declare parameter_source=method_descriptor"
            )

    assert not violations, (
        "FL peer_context_policy config는 exchange mechanism만 표현한다. FedMatch의 "
        "num_helpers/h_interval 같은 원본 기본값은 method package와 descriptor가 "
        "소유한다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
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


def test_agent_training_backend_old_path_is_not_reintroduced() -> None:
    package_root = AGENT_SRC / "services" / "training" / "backends" / "training"
    violations = [
        _relative_repo_path(path) for path in _iter_python_files(package_root)
    ]

    assert not violations, (
        "agent training backend old path는 재도입하지 않는다. concrete local update "
        "backend와 registry는 methods/adaptation/이 소유한다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_agent_does_not_own_pseudo_label_acceptance_policy_modules() -> None:
    package_root = AGENT_SRC / "services" / "training" / "acceptance_policies"
    violations = [
        _relative_repo_path(path) for path in _iter_python_files(package_root)
    ]

    assert not violations, (
        "pseudo-label acceptance/selection 정책 의미는 methods/ssl/hooks가 소유한다. "
        "agent는 methods-owned hook/spec을 local candidate/context에 연결하는 "
        "runtime adapter만 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_agent_does_not_own_privacy_guard_modules() -> None:
    package_root = AGENT_SRC / "services" / "training" / "execution" / "privacy_guards"
    violations = [
        _relative_repo_path(path) for path in _iter_python_files(package_root)
    ]

    assert not violations, (
        "privacy guard 정책과 adapter-family별 clipping 계산은 "
        "methods/adaptation/privacy_guards가 소유한다. agent는 selected guard를 "
        "local update 실행 흐름에 연결만 한다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_privacy_guards_stay_runtime_and_objective_agnostic() -> None:
    violations = _find_forbidden_imports(
        root=METHODS_SRC / "adaptation" / "privacy_guards",
        forbidden_prefixes=(
            "agent.src",
            "main_server.src",
            "scripts",
            "methods.ssl",
            "methods.federated_ssl.fedmatch",
            "methods.adaptation.peft_text_classifier",
            "methods.adaptation.text_classifier",
            "methods.adaptation.lora_classifier",
            "methods.adaptation.query_text_views",
        ),
    )

    assert not violations, (
        "privacy_guards는 shared adapter update 보호 policy만 소유한다. runtime, "
        "training objective, SSL method 의미를 import하지 않는다.\n"
        f"{_format_violations(violations)}"
    )


def test_agent_scoring_backends_do_not_keep_adapter_family_modules() -> None:
    package_root = AGENT_SRC / "services" / "inference" / "scoring_backends"
    forbidden_fragments = ("classifier_head", "diagonal_scale", "lora_classifier")
    violations = [
        _relative_repo_path(path)
        for path in _iter_python_files(package_root)
        if any(fragment in path.stem for fragment in forbidden_fragments)
    ]

    assert not violations, (
        "adapter-family별 scoring core는 methods/adaptation/<family>가 소유한다. "
        "agent scoring backend package에는 generic bridge와 local runtime glue만 "
        "둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_server_update_materialization_dispatcher_stays_family_agnostic() -> None:
    dispatcher_path = METHODS_SRC / "adaptation" / "server_update_materialization.py"
    imports = _collect_absolute_imports(dispatcher_path)
    forbidden_imports = {
        "shared.src.contracts.adapter_contract_families.classifier_head",
        "shared.src.contracts.adapter_contract_families.diagonal_scale",
        "shared.src.contracts.adapter_contract_families.lora_classifier",
    }
    violations = sorted(imports & forbidden_imports)
    source = dispatcher_path.read_text(encoding="utf-8")

    assert not violations, (
        "server update materialization dispatcher는 adapter family별 payload "
        "contract를 직접 알지 않는다. family-specific preflight는 "
        "methods/adaptation/<family>/server_preflight.py에 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )
    assert "agent-local://" not in source, (
        "agent-local artifact ref 정책은 dispatcher가 아니라 해당 adapter family가 "
        "소유한다."
    )
    assert "peft_classifier" not in source, (
        "dispatcher는 PEFT-classifier family 이름도 하드코딩하지 않는다. "
        "패키지 경로 alias는 adapter_family_modules.py가 소유한다."
    )


def test_server_update_compatibility_dispatcher_stays_family_agnostic() -> None:
    dispatcher_path = METHODS_SRC / "adaptation" / "server_update_compatibility.py"
    imports = _collect_absolute_imports(dispatcher_path)
    forbidden_imports = {
        "shared.src.contracts.adapter_contract_families.classifier_head",
        "shared.src.contracts.adapter_contract_families.diagonal_scale",
        "shared.src.contracts.adapter_contract_families.lora_classifier",
    }
    source = dispatcher_path.read_text(encoding="utf-8")

    assert not sorted(imports & forbidden_imports), (
        "server update compatibility dispatcher는 adapter family별 payload "
        "contract를 직접 알지 않는다."
    )
    assert "lora_classifier" not in source
    assert "peft_classifier" not in source


def test_runtime_objective_compatibility_dispatcher_stays_family_agnostic() -> None:
    dispatcher_path = METHODS_SRC / "adaptation" / "runtime_objective_compatibility.py"
    imports = _collect_absolute_imports(dispatcher_path)
    forbidden_imports = {
        "shared.src.contracts.adapter_contract_families.classifier_head",
        "shared.src.contracts.adapter_contract_families.diagonal_scale",
        "shared.src.contracts.adapter_contract_families.lora_classifier",
    }
    violations = sorted(imports & forbidden_imports)
    source = dispatcher_path.read_text(encoding="utf-8")

    assert not violations, (
        "runtime/objective compatibility dispatcher는 adapter family별 payload "
        "contract를 직접 알지 않는다. family-specific 검증은 "
        "methods/adaptation/<family>/runtime_compatibility.py에 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )
    assert "lora_classifier" not in source, (
        "dispatcher는 LoRA-classifier family 이름을 하드코딩하지 않는다."
    )
    assert "peft_classifier" not in source, (
        "dispatcher는 PEFT-classifier family 이름도 하드코딩하지 않는다. "
        "패키지 경로 alias는 adapter_family_modules.py가 소유한다."
    )


def test_fl_simulation_runtime_compatibility_adapter_is_family_agnostic() -> None:
    path = (
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "adapters"
        / "runtime_compatibility.py"
    )
    imports = _collect_absolute_imports(path)
    source = path.read_text(encoding="utf-8")

    assert "methods.adaptation.lora_classifier" not in imports, (
        "FL simulation runtime compatibility adapter는 LoRA 구현을 직접 import하지 "
        "않고 methods-owned dispatcher만 호출한다."
    )
    assert "lora_classifier" not in source, (
        "FL simulation runtime compatibility adapter는 adapter family literal로 "
        "분기하지 않는다."
    )


def test_federated_ssl_server_update_dispatcher_stays_family_agnostic() -> None:
    dispatcher_path = METHODS_SRC / "adaptation" / "federated_ssl_server_update.py"
    imports = _collect_absolute_imports(dispatcher_path)
    forbidden_imports = {
        "shared.src.contracts.adapter_contract_families.classifier_head",
        "shared.src.contracts.adapter_contract_families.diagonal_scale",
        "shared.src.contracts.adapter_contract_families.lora_classifier",
    }
    source = dispatcher_path.read_text(encoding="utf-8")

    assert not sorted(imports & forbidden_imports), (
        "FL SSL server update dispatcher는 adapter family별 payload contract를 "
        "직접 알지 않는다. family-specific backend 해석은 "
        "methods/adaptation/<family>/federated_ssl/server_update_policy.py가 "
        "소유한다."
    )
    assert "lora_classifier" not in source, (
        "FL SSL server update dispatcher는 LoRA-classifier family 이름을 "
        "하드코딩하지 않는다."
    )
    assert "peft_classifier" not in source, (
        "FL SSL server update dispatcher는 PEFT-classifier family 이름도 "
        "하드코딩하지 않는다."
    )


def test_lora_classifier_does_not_keep_server_preflight_shims() -> None:
    package_root = METHODS_SRC / "adaptation" / "lora_classifier"
    forbidden_paths = (
        package_root / "server_update_compatibility.py",
        package_root / "server_update_materialization.py",
    )
    violations = [
        _relative_repo_path(path) for path in forbidden_paths if path.exists()
    ]

    assert not violations, (
        "삭제된 methods/adaptation/lora_classifier package 아래에 dispatcher "
        "convention용 재-export shim을 다시 만들지 않는다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_lora_classifier_update_package_does_not_keep_one_use_helper_files() -> None:
    package_root = METHODS_SRC / "adaptation" / "lora_classifier" / "update"
    forbidden_paths = (
        package_root / "artifact_refs.py",
        package_root / "metrics.py",
        package_root / "row_extractor.py",
    )
    violations = [
        _relative_repo_path(path) for path in forbidden_paths if path.exists()
    ]

    assert not violations, (
        "삭제된 methods/adaptation/lora_classifier/update package 아래에 단일 "
        "사용처 helper 파일을 다시 만들지 않는다. 새 구현은 "
        "methods/adaptation/peft_text_classifier/** owner 경계에 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
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


def test_prototype_threshold_sweep_runner_splits_eval_selection_and_write() -> None:
    runner_path = PROTOTYPE_STRATEGY_SRC / "sweep.py"
    evaluator_path = PROTOTYPE_STRATEGY_SRC / "threshold_policy_evaluator.py"
    selection_path = PROTOTYPE_STRATEGY_SRC / "threshold_selection.py"
    writer_path = PROTOTYPE_STRATEGY_SRC / "threshold_artifact_writer.py"
    required_files = (evaluator_path, selection_path, writer_path)
    runner_source = runner_path.read_text(encoding="utf-8")
    evaluator_source = evaluator_path.read_text(encoding="utf-8")
    runner_forbidden_snippets = (
        "policy.build_evaluations(",
        "score_embeddings(",
        "dump_json(",
        ".write_text(",
        ".mkdir(",
        "_confidence_threshold_or_floor",
    )
    evaluator_forbidden_snippets = (
        "dump_json(",
        ".write_text(",
        ".mkdir(",
    )
    missing_files = [
        _relative_repo_path(path) for path in required_files if not path.exists()
    ]
    runner_violations = [
        snippet for snippet in runner_forbidden_snippets if snippet in runner_source
    ]
    evaluator_violations = [
        snippet
        for snippet in evaluator_forbidden_snippets
        if snippet in evaluator_source
    ]

    assert not missing_files, (
        "prototype threshold sweep는 policy 평가, selection policy, artifact writer를 "
        "전용 module로 분리한다.\n"
        f"{chr(10).join(f'- {path}' for path in missing_files)}"
    )
    assert not runner_violations, (
        "ThresholdPolicyExperimentRunner는 orchestration만 맡는다. threshold 후보 "
        "평가, 선택 정렬 기준, JSON artifact 저장은 전용 module이 맡는다.\n"
        f"violations={runner_violations}"
    )
    assert not evaluator_violations, (
        "threshold_policy_evaluator.py는 후보 평가만 맡는다. JSON 저장과 directory "
        "생성은 threshold_artifact_writer.py가 맡는다.\n"
        f"violations={evaluator_violations}"
    )


def test_prototype_strategy_scoring_does_not_use_runtime_fallback_profile() -> None:
    path = (
        SCRIPTS_SRC
        / "experiments"
        / "prototype_analysis"
        / "prototype_strategy"
        / "scoring.py"
    )
    imports = _collect_absolute_imports(path)

    assert "methods.federated_ssl.runtime_fallbacks" not in imports, (
        "prototype strategy scorer 기본값은 prototype 실험 축의 로컬 상수가 "
        "소유한다. FL SSL API/runtime fallback profile을 실험 기본값 "
        "source-of-truth처럼 import하지 않는다."
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
        "FedAvg strategy wiring 파일은 family별 aggregation/spec을 소유하지 않는다. "
        "family 상세는 methods/adaptation/<family>/aggregation/fedavg.py에 둔다.\n"
        f"violations={violations}"
    )


def test_adapter_family_fedavg_modules_live_under_aggregation_package() -> None:
    family_roots = (
        METHODS_SRC / "adaptation" / "diagonal_scale",
        METHODS_SRC / "adaptation" / "classifier_head",
        METHODS_SRC / "adaptation" / "lora_classifier",
    )
    forbidden_paths = [
        _relative_repo_path(path)
        for family_root in family_roots
        for path in (
            family_root / "fedavg.py",
            family_root / "fedavg_projection.py",
        )
        if path.exists()
    ]

    assert not forbidden_paths, (
        "adapter family별 FedAvg core/projection은 root 수평 파일이 아니라 "
        "methods/adaptation/<family>/aggregation/fedavg.py에 모은다.\n"
        f"{chr(10).join(f'- {path}' for path in forbidden_paths)}"
    )


def test_fedavg_aggregation_package_stays_generic() -> None:
    package_root = METHODS_SRC / "federated" / "aggregation" / "fedavg"
    allowed_files = {
        package_root / "__init__.py",
        package_root / "strategy.py",
        package_root / "update_metrics.py",
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


def test_partitioned_delta_average_does_not_create_generic_backend_folder() -> None:
    package_root = METHODS_SRC / "federated" / "aggregation" / "partitioned_fedavg"
    violations = (
        []
        if not package_root.exists()
        else [_relative_repo_path(path) for path in _iter_python_files(package_root)]
    )

    assert not violations, (
        "partitioned delta 평균은 adapter-family payload 해석이 먼저 필요한 backend다. "
        "registry convention만 만족시키는 methods/federated/aggregation/partitioned_* "
        "얇은 package를 만들지 않는다.\n"
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
