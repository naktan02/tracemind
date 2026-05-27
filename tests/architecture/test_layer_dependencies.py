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
QUERY_PEFT_SSL_SRC = SCRIPTS_SRC / "experiments" / "query_peft_ssl"
QUERY_PEFT_SSL_IO_SRC = QUERY_PEFT_SSL_SRC / "io"
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
TEST_FIXTURES_SRC = REPO_ROOT / "tests" / "fixtures"
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


def test_fl_local_update_profiles_do_not_keep_lora_classifier_leaf() -> None:
    profile_root = CONF_SRC / "strategy_axes" / "fl" / "local_update_profile"
    forbidden_path = profile_root / "lora_pseudo_label_v1.yaml"
    assert not forbidden_path.exists(), (
        "active FL local update profile leaf는 peft_pseudo_label_v1을 사용한다. "
        "lora_pseudo_label_v1은 old-run artifact/report reader compatibility "
        "표면으로만 남기고 Hydra 실행 profile로 되살리지 않는다."
    )


def test_fl_entrypoint_does_not_embed_lora_classifier_runtime_scope() -> None:
    path = CONF_SRC / "entrypoints" / "fl_ssl" / "run_federated_simulation.yaml"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "round_runtime.lora_classifier",
        "round_runtime.peft_classifier",
        "training_task.objective.lora_classifier",
        "artifact_ref_prefix: agent-local://lora_classifier",
        "lora_pseudo_label_v1",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]
    assert not violations, (
        "FL simulation entrypoint는 현재 실행 조합만 소유한다. lora_classifier "
        "runtime scope와 legacy profile leaf는 old-run reader compatibility에 "
        "격리하고 root Hydra entrypoint에 다시 복제하지 않는다.\n"
        f"{chr(10).join(f'- {snippet}' for snippet in violations)}"
    )


def test_fl_simulation_runtime_model_does_not_embed_lora_classifier_scope() -> None:
    checked_paths = (
        SCRIPTS_SRC / "experiments" / "fl_ssl" / "federated_simulation" / "models.py",
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "config_request.py",
        SCRIPTS_RUNTIME_ADAPTER_SRC
        / "federated_server"
        / "peft_encoder_round_runtime.py",
        METHODS_SRC / "adaptation" / "peft_text_classifier" / "runtime_family.py",
        METHODS_SRC / "adaptation" / "peft_text_classifier" / "resource_cache.py",
    )
    forbidden_snippets = (
        'round_runtime_payloads.get("lora_classifier")',
        'round_runtime_mapping.get("lora_classifier")',
        "lora_classifier: FederatedPeftEncoderRuntimeConfig",
        "round_runtime_config.lora_classifier",
        "or round_runtime_config.lora_classifier",
        'adapter_family_name == "lora_classifier"',
        "LORA_CLASSIFIER_ADAPTER_KIND,",
        "PEFT_ENCODER_LEGACY_RESOURCE_CACHE_NAMESPACE",
        "peft_encoder_legacy_resource_cache_prefix",
    )
    violations: list[tuple[Path, str]] = []
    for path in checked_paths:
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in source:
                violations.append((_relative_repo_path(path), snippet))

    assert not violations, (
        "active FL simulation runtime은 peft_classifier bootstrap scope만 연다. "
        "v1 lora_classifier는 shared contract/old artifact reader compatibility "
        "표면으로만 남기고, runtime model/payload builder에 다시 직접 열지 않는다.\n"
        f"{chr(10).join(f'- {path}: {snippet}' for path, snippet in violations)}"
    )


def test_fl_round_runtime_model_uses_generic_update_family_payloads() -> None:
    checked_paths = (
        SCRIPTS_SRC / "experiments" / "fl_ssl" / "federated_simulation" / "models.py",
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "config_request.py",
        SCRIPTS_RUNTIME_ADAPTER_SRC
        / "federated_agent"
        / "peft_encoder_method_owned_client_round.py",
        SCRIPTS_RUNTIME_ADAPTER_SRC
        / "federated_agent"
        / "peft_encoder_query_ssl_client_round.py",
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "adapters"
        / "runtime_compatibility.py",
        METHODS_SRC / "adaptation" / "peft_text_classifier" / "runtime_family.py",
        SCRIPTS_RUNTIME_ADAPTER_SRC
        / "federated_server"
        / "peft_encoder_round_runtime.py",
    )
    forbidden_snippets = (
        "peft_classifier: FederatedPeftEncoderRuntimeConfig",
        'round_runtime_payloads.get("peft_classifier")',
        'round_runtime_mapping.get("peft_classifier")',
        "round_runtime_mapping.get(PEFT_CLASSIFIER_ADAPTER_KIND)",
        "round_runtime.peft_classifier",
        "round_runtime.lora_classifier",
        "runtime_payload_for_adapter_family",
    )
    violations: list[tuple[Path, str]] = []
    for path in checked_paths:
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in source:
                violations.append((_relative_repo_path(path), snippet))

    assert not violations, (
        "FL round runtime model은 update family별 payload를 generic map으로 보관한다. "
        "새 update family 추가 때 scripts model/config_request에 family-specific "
        "field를 추가하지 않는다.\n"
        f"{chr(10).join(f'- {path}: {snippet}' for path, snippet in violations)}"
    )


def test_fl_simulation_server_aggregate_namespace_uses_update_family() -> None:
    checked_paths = (
        SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_server" / "runtime.py",
        SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_server" / "aggregation_artifacts.py",
        SCRIPTS_RUNTIME_ADAPTER_SRC
        / "federated_server"
        / "peft_encoder_server_step.py",
        REPO_ROOT / "tests" / "unit" / "test_run_federated_simulation.py",
    )
    forbidden_snippets = (
        "server-aggregate://{adapter_family_name}",
        "server-aggregate://peft_classifier",
        '/ "peft_classifier"\n        / "sim_rev_',
        "adapter_family_name=str(active.adapter_state.adapter_kind)",
    )
    violations: list[tuple[Path, str]] = []
    for path in checked_paths:
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in source:
                violations.append((_relative_repo_path(path), snippet))

    assert not violations, (
        "새 FL simulation server aggregate artifact namespace는 update_family_name을 "
        "사용한다. adapter_kind 기반 namespace는 old-run reader/operations fixture에만 "
        "남긴다.\n"
        f"{chr(10).join(f'- {path}: {snippet}' for path, snippet in violations)}"
    )


def test_fl_simulation_unit_tests_use_active_peft_payload_surface() -> None:
    source = (
        REPO_ROOT / "tests" / "unit" / "test_run_federated_simulation.py"
    ).read_text(encoding="utf-8")
    forbidden_snippets = (
        "make_lora_classifier_delta_payload",
        "LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT",
        'payload_kind="lora_classifier_materialized_state.v1"',
        'adapter_family_name="diagonal_scale"',
        'update_family_name="diagonal_scale"',
        'training_backend_name="diagonal_scale_heuristic"',
        'privacy_guard_name="diagonal_scale_clip_only"',
        'scorer_backend_name="diagonal_scale_logits"',
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "FL simulation unit fixture는 active PEFT-classifier payload surface를 "
        "검증한다. v1 lora_classifier/diagonal_scale payload 직접 생성은 shared "
        "contract compatibility 테스트로 격리한다.\n"
        f"{chr(10).join(f'- {snippet}' for snippet in violations)}"
    )


def test_scripts_runtime_bridges_use_peft_config_type_names() -> None:
    checked_paths = (
        SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_agent" / "local_training.py",
        SCRIPTS_RUNTIME_ADAPTER_SRC
        / "federated_server"
        / "peft_encoder_round_runtime.py",
    )
    violations = [
        _relative_repo_path(path)
        for path in checked_paths
        if "LoraClassifierTrainingBackendConfig" in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "scripts runtime bridge는 active PEFT encoder config type 이름을 사용한다. "
        "v1 LoraClassifierTrainingBackendConfig 이름은 methods/shared compatibility "
        "경계에만 남긴다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_peft_config_class_owns_canonical_defaults() -> None:
    path = PEFT_TEXT_CLASSIFIER_SRC / "config.py"
    source = path.read_text(encoding="utf-8")
    required_snippets = (
        "class PeftEncoderTrainingBackendConfig:",
        'artifact_ref_prefix: str = "agent-local://peft_classifier"',
        "payload_adapter_kind: str = PEFT_CLASSIFIER_PAYLOAD_ADAPTER_KIND",
        "PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL",
        "PEFT_ENCODER_DELTA_FORMAT_INLINE",
        "PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED",
        "class LoraClassifierTrainingBackendConfig(PeftEncoderTrainingBackendConfig):",
        'artifact_ref_prefix: str = "agent-local://lora_classifier"',
        "payload_adapter_kind: str = LORA_CLASSIFIER_PAYLOAD_ADAPTER_KIND",
        "def build_legacy_lora_classifier_training_backend_config(",
    )
    missing = [snippet for snippet in required_snippets if snippet not in source]
    forbidden_snippets = (
        "PeftClassifierTrainingBackendConfig = LoraClassifierTrainingBackendConfig",
        "PeftEncoderTrainingBackendConfig = LoraClassifierTrainingBackendConfig",
        "def build_lora_classifier_training_backend_config(",
        "LORA_CLASSIFIER_DELTA_FORMAT_AGENT_LOCAL",
        "LORA_CLASSIFIER_DELTA_FORMAT_INLINE",
        "LORA_CLASSIFIER_DELTA_FORMAT_SERVER_UPLOADED",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not missing and not violations, (
        "PEFT encoder config가 canonical class/default를 소유하고, "
        "LoraClassifierTrainingBackendConfig는 v1 compatibility subclass로만 "
        "남아야 한다.\n"
        f"missing={missing}\nviolations={violations}"
    )


def test_peft_training_backend_marks_legacy_lora_factories() -> None:
    path = PEFT_TEXT_CLASSIFIER_SRC / "training_backend.py"
    source = path.read_text(encoding="utf-8")
    required_snippets = (
        "def from_legacy_lora_objective_config(",
        "def build_legacy_lora_classifier_training_backend(",
    )
    forbidden_snippets = (
        "def from_objective_config(",
        "def build_lora_classifier_training_backend(",
    )
    missing = [snippet for snippet in required_snippets if snippet not in source]
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not missing and not violations, (
        "PEFT training backend의 v1 lora_classifier_trainer 경로는 legacy "
        "compatibility factory 이름으로 드러나야 한다. active PEFT factory처럼 "
        "보이는 generic/lora 이름을 다시 만들지 않는다.\n"
        f"missing={missing}\nviolations={violations}"
    )


def test_local_update_registry_does_not_embed_peft_backend_override() -> None:
    path = METHODS_SRC / "adaptation" / "local_update_registry.py"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "_TRAINING_BACKEND_MODULE_OVERRIDES",
        '"methods.adaptation.peft_text_classifier.training_backend"',
        '"peft_classifier_trainer":',
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "local update registry는 concrete backend module override를 직접 소유하지 "
        "않는다. 먼저 name convention을 시도하고, 필요하면 adaptation package scan으로 "
        "구현 옆 decorator 등록을 로드한다.\n"
        f"{chr(10).join(f'- {snippet}' for snippet in violations)}"
    )


def test_adapter_family_module_resolver_does_not_embed_concrete_family_map() -> None:
    path = METHODS_SRC / "adaptation" / "adapter_family_modules.py"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        '"classifier_head":',
        '"lora_classifier":',
        '"peft_classifier":',
        '"methods.adaptation.peft_text_classifier"',
        '"methods.classification.linear_head"',
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "adapter family module resolver는 concrete family alias table을 소유하지 "
        "않는다. alias 선언은 구현 owner 옆 adapter_family_module manifest에 둔다.\n"
        f"{chr(10).join(f'- {snippet}' for snippet in violations)}"
    )


def test_peft_method_modules_use_canonical_config_type_name() -> None:
    allowed_paths = {
        Path("methods/adaptation/peft_text_classifier/config.py"),
    }
    checked_roots = (
        PEFT_TEXT_CLASSIFIER_SRC,
        METHODS_FEDERATED_SSL_SRC / "fedmatch",
    )
    violations = [
        _relative_repo_path(path)
        for root in checked_roots
        for path in _iter_python_files(root)
        if _relative_repo_path(path) not in allowed_paths
        and "LoraClassifierTrainingBackendConfig" in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "PEFT text-classifier/FedMatch active method modules는 canonical "
        "PeftEncoderTrainingBackendConfig type 이름을 사용한다. legacy "
        "LoraClassifierTrainingBackendConfig는 config.py의 v1 compatibility "
        "subclass/builder에만 남긴다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_peft_method_modules_use_canonical_model_type_name() -> None:
    forbidden_snippets = (
        "LoraTextClassifier",
        "LoraClassifierModelRuntimeConfig",
        "build_lora_text_classifier_from_config",
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for root in (PEFT_TEXT_CLASSIFIER_SRC, METHODS_FEDERATED_SSL_SRC / "fedmatch")
        for path in _iter_python_files(root)
        for snippet in forbidden_snippets
        if snippet in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "PEFT text-classifier/FedMatch active method modules는 model/runtime type도 "
        "PeftEncoderTextClassifier와 PeftEncoderModelRuntimeConfig 이름을 사용한다. "
        "LoRA는 PEFT adapter mechanism 이름으로만 남긴다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_partitioned_peft_execution_primitive_uses_adapter_classifier_names() -> None:
    legacy_test_path = (
        REPO_ROOT
        / "tests"
        / "unit"
        / "test_methods_fedmatch_lora_partitioned_training.py"
    )
    active_test_path = (
        REPO_ROOT
        / "tests"
        / "unit"
        / "test_methods_fedmatch_peft_partitioned_training.py"
    )
    checked_paths = (
        PEFT_TEXT_CLASSIFIER_SRC / "federated_ssl" / "partitioned" / "training_loop.py",
        PEFT_TEXT_CLASSIFIER_SRC
        / "federated_ssl"
        / "partitioned_objective_training.py",
        active_test_path,
    )
    forbidden_snippets = (
        "PartitionedLoraStepResult",
        "PartitionedLoraTrainingResult",
        "train_partitioned_lora_classifier",
        "run_partitioned_lora_classifier_step",
        "FedMatch LoRA-classifier partitioned optimizer step tests",
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for path in checked_paths
        for snippet in forbidden_snippets
        if snippet in path.read_text(encoding="utf-8")
    ]

    assert (
        active_test_path.exists() and not legacy_test_path.exists() and not violations
    ), (
        "partitioned PEFT execution primitive는 adapter-classifier 실행 이름을 "
        "사용한다. LoRA 이름은 adapter mechanism parameter나 v1 payload projection "
        "표면에만 남긴다.\n"
        f"legacy_exists={legacy_test_path.exists()}\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_scripts_use_query_peft_ssl_harness_package_path() -> None:
    legacy_root = SCRIPTS_SRC / "experiments" / "query_lora_ssl"
    checked_roots = (SCRIPTS_SRC, REPO_ROOT / "tests")
    forbidden_snippets = (
        "scripts.experiments." + "query_lora_ssl",
        "scripts/experiments/" + "query_lora_ssl",
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for root in checked_roots
        for path in _iter_python_files(root)
        if path != Path(__file__).resolve()
        for snippet in forbidden_snippets
        if snippet in path.read_text(encoding="utf-8")
    ]

    assert (
        QUERY_PEFT_SSL_SRC.is_dir() and not legacy_root.exists() and not violations
    ), (
        "중앙 Query SSL harness package 경로는 query_peft_ssl을 사용한다. "
        "LoRA는 PEFT adapter mechanism 또는 v1 artifact/contract 이름으로만 "
        "남기고, scripts package boundary 이름으로 재도입하지 않는다.\n"
        f"legacy_exists={legacy_root.exists()}\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_central_ssl_entrypoints_use_peft_classifier_names() -> None:
    expected_paths = (
        SCRIPTS_SRC
        / "experiments"
        / "central_ssl_control"
        / "train_peft_ssl_classifier.py",
        SCRIPTS_SRC
        / "experiments"
        / "central_ssl_control"
        / "train_peft_supervised_classifier.py",
        CONF_SRC
        / "entrypoints"
        / "central_ssl_control"
        / "train_peft_ssl_classifier.yaml",
        CONF_SRC
        / "entrypoints"
        / "central_ssl_control"
        / "train_peft_supervised_classifier.yaml",
    )
    legacy_paths = (
        SCRIPTS_SRC
        / "experiments"
        / "central_ssl_control"
        / "train_lora_ssl_classifier.py",
        SCRIPTS_SRC
        / "experiments"
        / "central_ssl_control"
        / "train_lora_supervised_classifier.py",
        CONF_SRC
        / "entrypoints"
        / "central_ssl_control"
        / "train_lora_ssl_classifier.yaml",
        CONF_SRC
        / "entrypoints"
        / "central_ssl_control"
        / "train_lora_supervised_classifier.yaml",
    )
    checked_paths = (
        SCRIPTS_SRC / "README.md",
        SCRIPTS_SRC / "experiments" / "README.md",
        SCRIPTS_SRC / "experiments" / "central_ssl_control" / "README.md",
        SCRIPTS_SRC / "experiments" / "query_peft_ssl" / "README.md",
    )
    forbidden_snippets = (
        "train_lora_ssl_classifier",
        "train_lora_supervised_classifier",
    )
    missing_paths = [
        _relative_repo_path(path) for path in expected_paths if not path.exists()
    ]
    legacy_existing_paths = [
        _relative_repo_path(path) for path in legacy_paths if path.exists()
    ]
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for path in checked_paths
        for snippet in forbidden_snippets
        if snippet in path.read_text(encoding="utf-8")
    ]

    assert not (missing_paths or legacy_existing_paths or violations), (
        "중앙 SSL 실행 entrypoint/config/README는 PEFT classifier scaffold 이름을 "
        "사용한다. v1 lora_classifier 이름은 artifact schema와 old-run reader "
        "compatibility 표면에만 남긴다.\n"
        f"missing={missing_paths}\n"
        f"legacy_existing={legacy_existing_paths}\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_central_peft_entrypoints_do_not_write_lora_named_artifact_roots() -> None:
    checked_paths = (
        CONF_SRC
        / "entrypoints"
        / "central_ssl_control"
        / "train_peft_ssl_classifier.yaml",
        CONF_SRC
        / "entrypoints"
        / "central_ssl_control"
        / "train_peft_supervised_classifier.yaml",
    )
    forbidden_snippets = (
        "lora_adapters",
        "lora_classifier_heads",
        "lora_pseudo_label",
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for path in checked_paths
        for snippet in forbidden_snippets
        if snippet in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "새 중앙 PEFT 실행 산출물 기본 root는 PEFT/query 의미로 이름 붙인다. "
        "LoRA는 adapter mechanism 또는 old artifact reader compatibility 표면에만 "
        "남긴다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_query_peft_ssl_harness_uses_peft_helper_names() -> None:
    forbidden_snippets = (
        "query_" + "lora",
        "Query" + "Lora",
        "run_query_ssl_" + "lora",
        "train_query_ssl_" + "lora",
        "Supervised" + "Lora",
        "Lora" + "Labeled",
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for path in _iter_python_files(QUERY_PEFT_SSL_SRC)
        for snippet in forbidden_snippets
        if snippet in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "query_peft_ssl harness 내부 helper/type 이름은 PEFT 기준을 사용한다. "
        "LoRA는 adapter mechanism이나 old-run artifact/entrypoint compatibility "
        "표면에만 남긴다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_peft_partition_delta_uses_canonical_internal_type_name() -> None:
    checked_roots = (
        PEFT_TEXT_CLASSIFIER_SRC,
        REPO_ROOT
        / "tests"
        / "unit"
        / "test_methods_fedmatch_peft_partitioned_training.py",
        REPO_ROOT / "tests" / "unit" / "test_federated_agent_runtime_adapters.py",
        REPO_ROOT / "tests" / "unit" / "test_peft_encoder_aggregation.py",
    )
    forbidden_snippets = (
        "LoraClassifierPartitionDelta",
        "build_lora_classifier_partition_delta_from_parameter_deltas",
        "project_adapter_classifier_delta_bundle_to_lora_partition_delta",
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for root in checked_roots
        for path in (_iter_python_files(root) if root.is_dir() else [root])
        for snippet in forbidden_snippets
        if snippet in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "PEFT encoder 내부 partition delta value object는 "
        "PeftEncoderPartitionDelta 이름을 사용한다. lora_classifier 이름은 "
        "shared v1 contract 또는 artifact schema 문자열에만 남긴다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_query_ssl_update_unit_test_uses_peft_encoder_surface() -> None:
    legacy_path = REPO_ROOT / "tests" / "unit" / "test_query_ssl_lora_update.py"
    active_path = REPO_ROOT / "tests" / "unit" / "test_query_ssl_peft_encoder_update.py"
    source = active_path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "LoraClassifierTrainingBackendConfig",
        "lora_classifier_trainer",
        "payload.lora_delta_artifact_ref",
        "payload.lora_parameter_deltas",
        "test_query_ssl_lora_update",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert active_path.exists() and not legacy_path.exists() and not violations, (
        "Query SSL update unit test는 active PEFT encoder update payload surface를 "
        "검증한다. v1 lora_classifier payload 검증은 shared contract compatibility "
        "테스트에 격리한다.\n"
        f"legacy_exists={legacy_path.exists()}\nviolations={violations}"
    )


def test_partitioned_model_builder_unit_test_uses_peft_encoder_surface() -> None:
    legacy_path = (
        REPO_ROOT / "tests" / "unit" / "test_partitioned_lora_model_builder.py"
    )
    active_path = (
        REPO_ROOT / "tests" / "unit" / "test_partitioned_peft_encoder_model_builder.py"
    )
    source = active_path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "LoraClassifierTrainingBackendConfig",
        "test_partitioned_lora_builder",
        "LoRA-classifier builder tests",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert active_path.exists() and not legacy_path.exists() and not violations, (
        "Partitioned model builder unit test는 active PEFT encoder builder surface를 "
        "검증한다. LoRA는 adapter parameter 예시나 v1 compatibility 테스트에만 "
        "남긴다.\n"
        f"legacy_exists={legacy_path.exists()}\nviolations={violations}"
    )


def test_peft_core_unit_tests_use_canonical_config_type() -> None:
    checked_paths = (
        REPO_ROOT / "tests" / "unit" / "test_peft_encoder_runtime_resources.py",
        REPO_ROOT / "tests" / "unit" / "test_peft_encoder_training_core.py",
        REPO_ROOT / "tests" / "unit" / "test_run_federated_simulation.py",
    )
    violations = [
        _relative_repo_path(path)
        for path in checked_paths
        if "LoraClassifierTrainingBackendConfig" in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "PEFT encoder core/resource unit tests는 canonical "
        "PeftEncoderTrainingBackendConfig를 사용한다. v1 config subclass는 "
        "legacy payload compatibility 테스트에서만 직접 사용한다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_peft_training_primitives_unit_test_uses_peft_encoder_surface() -> None:
    legacy_path = REPO_ROOT / "tests" / "unit" / "test_lora_training_primitives.py"
    active_path = (
        REPO_ROOT / "tests" / "unit" / "test_peft_encoder_training_primitives.py"
    )
    source = active_path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "LoRA-classifier training primitive",
        "test_lora_training",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert active_path.exists() and not legacy_path.exists() and not violations, (
        "PEFT training primitive unit test는 active PEFT encoder surface를 "
        "검증한다. LoRA는 adapter mechanism이나 v1 compatibility payload 테스트에만 "
        "남긴다.\n"
        f"legacy_exists={legacy_path.exists()}\nviolations={violations}"
    )


def test_federated_agent_runtime_adapter_unit_tests_name_active_peft_surface() -> None:
    path = REPO_ROOT / "tests" / "unit" / "test_federated_agent_runtime_adapters.py"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "test_query_ssl_lora_local_training",
        "test_query_ssl_lora_delta_materialization",
        "test_lora_classifier_base_parameters",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]
    required_snippets = (
        "test_query_ssl_peft_encoder_local_training_resolves_selected_ssl_algorithm",
        "make_peft_classifier_state_payload",
        "make_peft_classifier_delta_payload",
        "test_upload_agent_local_lora_v1_update_materializes_server_owned_refs",
    )
    missing = [snippet for snippet in required_snippets if snippet not in source]

    assert not violations and not missing, (
        "federated agent runtime adapter unit test는 active 경로를 PEFT encoder로 "
        "부르고, v1 LoRA payload compatibility는 이름에 v1을 드러낸다.\n"
        f"violations={violations}\nmissing={missing}"
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


def test_runtime_fallback_profile_does_not_import_adapter_implementation() -> None:
    path = METHODS_FEDERATED_SSL_SRC / "runtime_fallbacks.py"
    imports = _collect_absolute_imports(path)
    source = path.read_text(encoding="utf-8")

    assert "methods.adaptation.diagonal_scale.config" not in imports, (
        "runtime fallback profile은 legacy compatibility 값을 소유하되 "
        "diagonal_scale 구현 config를 import하지 않는다. fallback이 남아 있어도 "
        "adapter implementation과 runtime default를 강결합하지 않는다."
    )
    forbidden_snippets = (
        "diagonal_scale_heuristic",
        "diagonal_scale_clip_only",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]
    assert not violations, (
        "runtime fallback은 삭제된 diagonal_scale 실행 구현을 기본값으로 선택하지 "
        "않는다. v1 diagonal_scale은 shared contract compatibility 표면에만 남긴다.\n"
        f"violations={violations}"
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
    source = path.read_text(encoding="utf-8")

    assert (
        "shared.src.contracts.adapter_contract_families.diagonal_scale" not in imports
    ), (
        "server runtime config는 shared diagonal_scale contract를 직접 import하지 "
        "않는다. legacy no-config fallback은 named runtime profile 값으로만 "
        "격리한다."
    )
    assert "legacy_diagonal_scale" not in source
    assert 'adapter_family_name="diagonal_scale"' not in source


def test_privacy_guards_do_not_register_removed_diagonal_scale_guard() -> None:
    path = METHODS_SRC / "adaptation" / "privacy_guards" / "clip_only.py"
    imports = _collect_absolute_imports(path)
    source = path.read_text(encoding="utf-8")

    assert (
        "shared.src.contracts.adapter_contract_families.diagonal_scale" not in imports
    )
    assert "diagonal_scale_clip_only" not in source


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


def test_fl_run_layout_stays_update_family_oriented() -> None:
    path = SCRIPTS_SRC / "experiments" / "fl_ssl" / "run_layout.py"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "training_task.objective.peft_classifier.proximal_mu",
        "training_task.objective.lora_classifier.proximal_mu",
        'f"round_runtime.{family}.peft_adapter_name"',
        "_resolve_adapter_runtime_slug",
        "_resolve_peft_adapter_kind",
        "peft_adapter_name",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "FL run layout은 active output path를 update_family_name 중심으로 만든다. "
        "FedProx 같은 local regularizer 표기는 objective payload에서 generic하게 "
        "읽고, classifier-family별 dotted path를 직접 소유하지 않는다.\n"
        f"{chr(10).join(f'- {snippet}' for snippet in violations)}"
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


def test_peft_projection_artifacts_do_not_keep_lora_classifier_aliases() -> None:
    script_shim_path = SCRIPTS_SRC / "experiments" / "lora_classifier_projection.py"
    projection_writer_path = (
        METHODS_SRC / "adaptation" / "peft_text_classifier" / "projection_artifacts.py"
    )
    source = projection_writer_path.read_text(encoding="utf-8")

    assert not script_shim_path.exists(), (
        "scripts는 PEFT projection writer의 legacy lora_classifier alias shim을 "
        "소유하지 않는다. 실행 표면은 conf-declared runtime adapter나 methods core를 "
        "직접 호출한다."
    )
    assert "write_lora_classifier_projection_artifacts" not in source, (
        "PEFT projection artifact writer는 canonical "
        "write_peft_encoder_projection_artifacts만 제공한다. legacy "
        "lora_classifier alias를 재도입하지 않는다."
    )


def test_peft_text_classifier_does_not_keep_legacy_lora_pass_through_aliases() -> None:
    alias_names = (
        "resolve_method_owned_lora_classifier_training_core",
        "_load_method_owned_lora_classifier_training_core",
        "run_method_owned_lora_classifier_training_core",
        "run_query_ssl_lora_classifier_training_core",
        "_build_lora_classifier_model",
        "evaluate_lora_classifier_state",
        "evaluate_lora_classifier_state_payload",
        "evaluate_lora_classifier_validation_payload",
        "require_lora_classifier_validation_backend",
        "require_lora_classifier_state",
        "lora_classifier_base_state_artifact_refs",
        "build_lora_classifier_training_row",
        "extract_lora_classifier_training_text",
        "build_lora_classifier_base_artifact_ref",
        "run_lora_classifier_supervised_seed_step_core",
        "compute_lora_classifier_partitioned_delta_average",
        "aggregate_lora_classifier_partitioned_delta_average",
        "require_lora_classifier_update_matches_active_state",
        "require_lora_classifier_update_is_server_materializable",
        "load_lora_classifier_base_parameters",
        "load_lora_classifier_base_partition_parameters",
        "_materialize_lora_classifier_base_parameters",
        "_materialize_lora_classifier_base_partition_parameters",
        "load_lora_classifier_base_parameters_into_model",
        "extract_lora_classifier_parameter_deltas",
        "lora_classifier_delta_l2_norm",
        "run_method_owned_lora_classifier_local_training",
        "run_query_ssl_lora_classifier_local_training",
        "build_query_ssl_lora_update_payload",
        "build_query_ssl_lora_client_metrics",
        "resolve_lora_classifier_federated_ssl_server_update_backend",
        "run_partitioned_lora_classifier_training_core",
        "build_lora_classifier_delta_update",
        "build_lora_classifier_helper_probability_provider",
        "build_lora_classifier_helper_provider_for_local_ssl_policy",
        "LoraClassifierHelperWeakProbabilityProvider",
        "build_lora_classifier_peer_client_snapshot",
        "compute_lora_classifier_probe_vector",
        "extract_lora_classifier_materialized_state",
        "compute_lora_classifier_fedavg",
        "aggregate_lora_classifier_fedavg",
        "validate_lora_classifier_update_matches_base",
        "build_lora_classifier_state_projection",
        "LoraClassifierMaterializedUpdate",
        "LoraClassifierMaterializedState",
        "compact_lora_classifier_materialized_state",
        "materialize_base_lora_classifier_state",
        "materialize_base_lora_classifier_partitioned_state",
        "materialize_lora_classifier_update",
        "materialize_lora_classifier_partitioned_update",
        "LoraTextClassifierFactory",
        "LoraClassifierPartitionRuntimeConfig",
        "PartitionedLoraTextClassifierBuildResult",
        "build_partitioned_lora_text_classifier_from_config",
        "LORA_CLASSIFIER_BASE_PARTITIONED_STATE_MATERIALIZER_NAME",
        "QuerySslLoraObjectiveRuntimeConfig",
        "LoraClassifierTrainerRuntimeConfig",
        "QuerySslLoraDeltaMaterialization",
        "QuerySslLoraDeltaMaterializer",
        "QuerySslLoraClientTrainingResult",
        "QuerySslLoraLocalTrainingRequest",
        "QuerySslLoraTrainingBackend",
        "SimulationInlineLoraClassifierTrainExecutor",
        "LoraClassifierSupervisedSeedStepResult",
        "FederatedLoraClassifierRuntimeConfig",
        "QuerySslLoraUpdateBuildResult",
        "LoraClassifierRuntimePayloadConfig",
        "LoraClassifierTrainingRow",
        "LoraClassifierTrainArtifacts",
        "LoraClassifierUpdateConfig",
        "LoraClassifierTrainExecutor",
        "require_peft_classifier_runtime_matches_objective",
        "PeftEncoderTextClassifier =",
        "upload_agent_local_lora_classifier_update",
        "server_owned_lora_classifier_update_artifact_byte_count",
        "LoraClassifierDeltaArtifactStore",
        "LoraClassifierDeltaMaterializer",
        "resolve_lora_classifier_label_schema",
        "build_lora_classifier_delta_from_rows",
        "build_lora_classifier_delta_payload_from_artifacts",
        "merge_partitioned_lora_classifier_deltas",
        "apply_lora_classifier_partition_delta_to_state",
        "apply_lora_classifier_partition_deltas_to_partitioned_state",
        "split_lora_classifier_state_by_residual_factor",
        "LoraClassifierFedAvgUpdate",
        "LoraClassifierFedAvgResult",
        "LoraClassifierStateProjection",
        "LoraClassifierPartitionedDeltaAverageUpdate",
    )
    checked_roots = (
        AGENT_SRC / "services" / "training" / "execution",
        METHODS_SRC / "adaptation" / "peft_text_classifier",
        METHODS_SRC / "federated_ssl" / "fedmatch",
        SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_agent",
        SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_server",
    )
    violations = [
        f"{_relative_repo_path(path)}: {alias_name}"
        for root in checked_roots
        for path in _iter_python_files(root)
        for alias_name in alias_names
        if alias_name in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "PEFT text classifier 내부 public surface는 canonical PEFT 이름을 사용한다. "
        "v1 lora_classifier 이름은 shared payload/schema, config compatibility, "
        "artifact field 의미처럼 실제 호환 경계에만 남긴다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_test_only_federated_ssl_fixture_stays_family_contract_agnostic() -> None:
    fixture_path = TEST_FIXTURES_SRC / "federated_ssl_dummy_method.py"
    imports = _collect_absolute_imports(fixture_path)
    forbidden_imports = {
        "shared.src.contracts.adapter_contract_families.classifier_head",
        "shared.src.contracts.adapter_contract_families.diagonal_scale",
        "shared.src.contracts.adapter_contract_families.lora_classifier",
    }
    forbidden_snippets = (
        "classifier_head",
        "diagonal_scale",
        "lora_classifier",
        "peft_classifier",
    )
    source = fixture_path.read_text(encoding="utf-8")
    snippet_violations = [
        snippet for snippet in forbidden_snippets if snippet in source
    ]

    assert not sorted(imports & forbidden_imports), (
        "test-only FL SSL method fixture는 특정 adapter-family payload contract를 "
        "import하지 않는다. fixture는 method extension seam만 검증해야 한다."
    )
    assert not snippet_violations, (
        "test-only FL SSL method fixture는 concrete adapter-family 이름을 "
        "하드코딩하지 않는다.\n"
        f"violations={snippet_violations}"
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


def test_diagonal_scale_no_longer_has_update_family_initialization_leaf() -> None:
    forbidden_paths = (
        CONF_SRC
        / "strategy_axes"
        / "trainable_state"
        / "update_family"
        / "diagonal_scale.yaml",
        METHODS_SRC / "adaptation" / "diagonal_scale",
    )
    existing_paths = _existing_non_cache_paths(forbidden_paths)

    assert not existing_paths, (
        "diagonal_scale는 target update-family 축이 아니다. v1 shared contract와 "
        "legacy fallback 값이 남아 있더라도 methods-level 구현 폴더나 "
        "trainable_state/update_family leaf를 다시 만들지 않는다.\n"
        f"{chr(10).join(f'- {path}' for path in existing_paths)}"
    )


def test_fl_round_e2e_does_not_exercise_removed_diagonal_scale_runtime() -> None:
    path = REPO_ROOT / "tests" / "integration" / "test_fl_round_e2e.py"
    imports = _collect_absolute_imports(path)
    source = path.read_text(encoding="utf-8")

    assert (
        "shared.src.contracts.adapter_contract_families.diagonal_scale" not in imports
    )
    assert "diagonal_scale" not in source, (
        "root FL round E2E는 현재 runtime family를 검증한다. diagonal_scale는 "
        "shared v1 contract compatibility 테스트에만 남기고, 서버/에이전트 "
        "lifecycle smoke의 실행 family로 되살리지 않는다."
    )


def test_agent_training_artifact_repository_uses_shared_update_payloads_only() -> None:
    path = (
        AGENT_SRC
        / "infrastructure"
        / "repositories"
        / "training_artifact_repository.py"
    )
    imports = _collect_absolute_imports(path)
    source = path.read_text(encoding="utf-8")

    assert (
        "shared.src.contracts.adapter_contract_families.diagonal_scale" not in imports
    )
    forbidden_snippets = (
        "VectorAdapterDelta",
        "save_vector_adapter_delta",
        "load_vector_adapter_delta",
        "delta_dir",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]
    assert not violations, (
        "agent artifact repository는 로컬 update 저장소 capability만 소유한다. "
        "삭제된 diagonal_scale/vector adapter compatibility alias를 agent runtime "
        "표면에 다시 열지 않는다.\n"
        f"{chr(10).join(f'- {snippet}' for snippet in violations)}"
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
        "패키지 경로 alias는 구현 owner 옆 adapter_family_module manifest가 소유한다."
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
        "패키지 경로 alias는 구현 owner 옆 adapter_family_module manifest가 소유한다."
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


def test_query_peft_run_artifacts_do_not_keep_writer_exporter_monolith() -> None:
    orchestrator_path = QUERY_PEFT_SSL_IO_SRC / "artifacts.py"
    expected_responsibility_files = (
        QUERY_PEFT_SSL_IO_SRC / "artifact_paths.py",
        QUERY_PEFT_SSL_IO_SRC / "artifact_writer.py",
        QUERY_PEFT_SSL_IO_SRC / "manifest_builder.py",
        QUERY_PEFT_SSL_IO_SRC / "model_artifact_exporter.py",
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
        "Query PEFT run artifact 저장은 경로, 모델 export, payload build, JSON write "
        "책임 파일로 나눈다.\n"
        f"{chr(10).join(f'- {path}' for path in missing_files)}"
    )
    assert not violations, (
        "artifacts.py는 public orchestration entrypoint만 유지한다. 파일 저장, "
        "JSON serialization, model export를 다시 한 함수에 모으지 않는다.\n"
        f"violations={violations}"
    )


def test_query_peft_teacher_pseudo_label_does_not_keep_exporter_monolith() -> None:
    legacy_exporter_path = QUERY_PEFT_SSL_IO_SRC / "teacher_pseudo_label_exporter.py"
    builder_path = QUERY_PEFT_SSL_IO_SRC / "teacher_pseudo_label_builder.py"
    writer_path = QUERY_PEFT_SSL_IO_SRC / "teacher_pseudo_label_artifact_writer.py"
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
