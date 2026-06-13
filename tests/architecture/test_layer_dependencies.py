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
CONF_FL_METHOD_DESCRIPTOR_SRC = REPO_ROOT / "conf" / "strategy_axes" / "fssl_method"
CONF_FL_UPDATE_PARTITION_POLICY_SRC = (
    REPO_ROOT / "conf" / "strategy_axes" / "fl_topology" / "update_partition"
)
CONF_FL_PEER_CONTEXT_POLICY_SRC = (
    REPO_ROOT / "conf" / "strategy_axes" / "fl_topology" / "peer_context"
)
AGENT_SRC = REPO_ROOT / "agent" / "src"
AGENT_CONF = REPO_ROOT / "agent" / "conf"
MAIN_SERVER_SRC = REPO_ROOT / "main_server" / "src"
APPS_SRC = REPO_ROOT / "apps"
SCRIPTS_SRC = REPO_ROOT / "scripts"
SCRIPTS_RUNTIME_ADAPTER_SRC = SCRIPTS_SRC / "runtime_adapters"
FL_SIMULATION_IO_SRC = (
    SCRIPTS_SRC / "experiments" / "fl_ssl" / "federated_simulation" / "io"
)
QUERY_SSL_TEXT_ENCODER_SRC = SCRIPTS_SRC / "support" / "query_ssl_text_encoder"
QUERY_SSL_TEXT_ENCODER_CONFIG_SRC = QUERY_SSL_TEXT_ENCODER_SRC / "config"
QUERY_SSL_TEXT_ENCODER_IO_SRC = QUERY_SSL_TEXT_ENCODER_SRC / "io"
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
METHODS_FEDERATED_SSL_SRC = METHODS_SRC / "federated_ssl"
METHODS_SSL_SRC = METHODS_SRC / "ssl"
METHODS_SSL_ALGORITHMS_SRC = METHODS_SSL_SRC / "algorithms"
PEFT_TEXT_ENCODER_SRC = METHODS_SRC / "adaptation" / "peft_text_encoder"
PEFT_TEXT_ENCODER_AGGREGATION_SRC = PEFT_TEXT_ENCODER_SRC / "aggregation"
LINEAR_HEAD_CLASSIFICATION_SRC = METHODS_SRC / "classification" / "linear_head"
PEFT_ADAPTERS_SRC = METHODS_SRC / "adaptation" / "peft_adapters"
LEGACY_AGENT_QUERY_TEXT_VIEWS_SRC = (
    AGENT_SRC / "services" / "training" / "query_text_views"
)

TEMPORARY_MAIN_SERVER_AGENT_IMPORT_EXCEPTIONS: set[Path] = set()
RUNTIME_LAYER_METHOD_NAME_FRAGMENTS = (
    "fedmatch",
    "fixmatch",
    "refixmatch",
    "remixmatch",
    "freematch",
    "flexmatch",
    "comatch",
    "dash",
    "simmatch",
    "mixmatch",
    "mixtext",
    "rdrop",
)
FL_SCRIPT_RUNTIME_ROOTS = (
    SCRIPTS_SRC / "experiments" / "fl_ssl",
    SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_agent",
    SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_server",
)
PAPER_METHOD_NAME_FRAGMENTS = ("fedmatch",)


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.py") if "__pycache__" not in path.parts
    )


def _iter_app_source_files(root: Path) -> list[Path]:
    suffixes = {".js", ".jsx", ".ts", ".tsx"}
    ignored_parts = {"dist", "node_modules", ".tmp"}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in suffixes
        and "src" in path.parts
        and not any(part in ignored_parts for part in path.parts)
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


def _find_forbidden_text_snippets(
    *,
    root: Path,
    snippets: tuple[str, ...],
) -> list[tuple[Path, str]]:
    violations: list[tuple[Path, str]] = []
    for path in _iter_app_source_files(root):
        source = path.read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet in source:
                violations.append((_relative_repo_path(path), snippet))
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


def test_central_query_ssl_support_does_not_import_agent_runtime() -> None:
    violations = _find_forbidden_imports(
        root=QUERY_SSL_TEXT_ENCODER_SRC,
        forbidden_prefixes=("agent.src", "main_server.src"),
    )
    assert not violations, (
        "central Query SSL support는 offline control adapter다. live agent/server "
        "runtime 재사용은 methods-owned local training surface를 통해 하고, "
        "agent.src/main_server.src를 직접 import하지 않는다.\n"
        f"{_format_violations(violations)}"
    )


def test_central_peft_ssl_uses_methods_local_training_request_surface() -> None:
    trainable_surface_path = (
        CONF_SRC
        / "strategy_axes"
        / "model_architecture"
        / "trainable_surface"
        / "peft_text_encoder.yaml"
    )
    runner_path = QUERY_SSL_TEXT_ENCODER_SRC / "runners" / "consistency.py"
    trainable_surface_source = trainable_surface_path.read_text(encoding="utf-8")
    runner_source = runner_path.read_text(encoding="utf-8")

    assert "run_query_ssl_peft_encoder_local_session" in trainable_surface_source
    assert "QuerySslPeftEncoderLocalSessionRequest" in runner_source
    assert "**local_session_request" not in runner_source, (
        "central SSL runner는 methods-owned dataclass request를 그대로 넘긴다. "
        "dict/kwargs surface로 되돌리면 central/agent/FL training 의미 drift를 "
        "테스트하기 어려워진다."
    )


def test_method_owned_fssl_uses_request_training_surface() -> None:
    surface_path = (
        PEFT_TEXT_ENCODER_SRC / "federated_ssl" / "method_training_surface.py"
    )
    agent_task_path = (
        AGENT_SRC
        / "services"
        / "training_runtime"
        / "query_ssl"
        / "method_request_builder.py"
    )
    fedmatch_descriptor_path = METHODS_FEDERATED_SSL_SRC / "fedmatch" / "descriptor.py"
    method_owned_path = (
        PEFT_TEXT_ENCODER_SRC / "federated_ssl" / "method_owned_training.py"
    )

    assert surface_path.exists()
    assert "FsslPeftEncoderMethodTrainingRequest" in surface_path.read_text(
        encoding="utf-8"
    )
    assert "FsslPeftEncoderMethodTrainingRequest" in agent_task_path.read_text(
        encoding="utf-8"
    )
    assert (
        "run_method_owned_peft_encoder_training_request"
        in fedmatch_descriptor_path.read_text(encoding="utf-8")
    )
    assert "return core(request)" in method_owned_path.read_text(encoding="utf-8")


def test_ssl_root_keeps_framework_surface_not_primitives() -> None:
    allowed_root_files = {
        "NEW_METHOD.md",
        "README.md",
        "__init__.py",
        "base.py",
        "model_capabilities.py",
        "registry.py",
        "state.py",
    }
    actual_root_files = {
        path.name
        for path in METHODS_SSL_SRC.iterdir()
        if path.is_file() and path.suffix in {".py", ".md"}
    }

    assert actual_root_files <= allowed_root_files, (
        "`methods/ssl` 바로 아래에는 framework surface만 둔다. "
        "여러 algorithm이 공유하는 tensor/module helper는 "
        "`methods/ssl/primitives/`에 둔다.\n"
        f"unexpected={sorted(actual_root_files - allowed_root_files)}"
    )
    assert (METHODS_SSL_SRC / "primitives" / "README.md").exists()


def test_ssl_algorithms_do_not_reuse_method_local_helper_packages() -> None:
    forbidden_prefixes = ("methods.ssl.algorithms.mixmatch",)
    violations = _find_forbidden_imports(
        root=METHODS_SSL_ALGORITHMS_SRC,
        forbidden_prefixes=forbidden_prefixes,
        ignored_roots=(METHODS_SSL_ALGORITHMS_SRC / "mixmatch",),
    )
    relative_violations = [
        _relative_repo_path(path)
        for path in _iter_python_files(METHODS_SSL_ALGORITHMS_SRC)
        if not path.is_relative_to(METHODS_SSL_ALGORITHMS_SRC / "mixmatch")
        and "from ..mixmatch" in path.read_text(encoding="utf-8")
    ]

    assert not violations and not relative_violations, (
        "다른 SSL algorithm이 MixMatch method-local helper를 직접 가져다 쓰지 않는다. "
        "공유 의미가 안정된 helper는 `methods/ssl/primitives/`로 승격한다.\n"
        f"{_format_violations(violations)}\n"
        f"relative_violations={relative_violations}"
    )


def test_shared_contracts_do_not_keep_central_payload_adapter_metadata_catalog() -> (
    None
):
    forbidden_path = SHARED_SRC / "contracts" / "payload_adapter_metadata.py"
    assert not forbidden_path.exists(), (
        "shared는 중앙 payload adapter metadata catalog를 소유하지 않는다. "
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
        "diagonal_scale_update",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "shared training envelope는 payload_format 문자열 필드만 소유한다. "
        "payload-adapter별 canonical/accepted format은 "
        "shared/src/contracts/adapter_contract_families/<family>.py가 소유한다.\n"
        f"violations={violations}"
    )


def test_shared_contract_readme_uses_active_adapter_kind_examples() -> None:
    path = SHARED_SRC / "contracts" / "README.md"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = ("예: `diagonal_scale`, `classifier_head`, `lora_classifier`",)
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "shared contract README의 일반 adapter_kind 예시는 active family 이름을 "
        "사용한다. diagonal_scale/lora_classifier는 legacy compatibility 설명에만 "
        "남긴다.\n"
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
        if path.exists() and snippet in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "shared adapter base/registry는 새 payload를 diagonal_scale로 암묵 해석하지 "
        "않는다. legacy vector_adapter schema compatibility만 명시적으로 허용한다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_classifier_head_v1_contract_stays_linear_head_explicit() -> None:
    contract_path = (
        SHARED_SRC / "contracts" / "adapter_contract_families" / "classifier_head.py"
    )
    update_family_dir = (
        CONF_SRC / "strategy_axes" / "model_architecture" / "update_family"
    )
    source = contract_path.read_text(encoding="utf-8")
    missing_snippets = [
        snippet
        for snippet in (
            'LINEAR_CLASSIFIER_HEAD_KIND = "linear"',
            'ClassifierHeadKind: TypeAlias = Literal["linear"]',
            "head_kind: ClassifierHeadKind",
        )
        if snippet not in source
    ]
    forbidden_leaf = update_family_dir / "classifier_head.yaml"

    assert not missing_snippets, (
        "classifier_head.v1은 generic classifier family 전체가 아니라 "
        "linear weight/bias payload contract다. contract에 head_kind=linear를 "
        "명시해 future MLP/projection head가 v1 shape에 섞이지 않게 한다.\n"
        f"missing={missing_snippets}"
    )
    assert not forbidden_leaf.exists(), (
        "config update_family leaf는 concrete 실행 단위여야 한다. 현재 "
        "classifier-head 실행 leaf는 linear_head.yaml이며, generic "
        "classifier_head.yaml placeholder를 만들지 않는다.\n"
        f"path={_relative_repo_path(forbidden_leaf)}"
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


def test_active_prototype_surface_is_removed_until_method_is_adopted() -> None:
    removed_paths = (
        METHODS_SRC / "prototype",
        SHARED_SRC / "contracts" / "prototype_contracts.py",
        SHARED_SRC / "contracts" / "prototype_build_state_contracts.py",
        SHARED_SRC / "services" / "prototypes",
        MAIN_SERVER_SRC / "services" / "federation" / "prototypes",
        SCRIPTS_SRC / "workflows" / "prototype_pack",
        CONF_SRC / "strategy_axes" / "prototype",
        CONF_SRC / "entrypoints" / "prototype_pack",
    )
    existing_paths = [
        _relative_repo_path(path) for path in removed_paths if path.exists()
    ]

    assert not existing_paths, (
        "prototype 방법론은 현재 active 연구/운영 surface가 아니다. 다시 도입할 때 "
        "methods/conf/runtime adapter를 한 번에 열고, 지금은 잔재를 두지 않는다.\n"
        f"{chr(10).join(f'- {path}' for path in existing_paths)}"
    )


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
            "methods.adaptation.peft_text_encoder",
            "methods.adaptation.peft_adapters",
            "shared.src.contracts.adapter_contract_families",
            "shared.src.domain.entities.training.shared_adapter_state",
            "shared.src.domain.entities.training.shared_adapter_update",
        ),
    )

    assert not violations, (
        "query_text_views는 query-domain row/view/token-batch 입력 glue만 "
        "소유한다. PEFT model composition, shared update payload, payload-adapter "
        "materialization은 각 canonical owner에 둔다.\n"
        f"{_format_violations(violations)}"
    )


def test_query_ssl_view_preparation_core_stays_in_methods_layer() -> None:
    legacy_script_path = QUERY_SSL_TEXT_ENCODER_SRC / "query_ssl" / "augmentation.py"
    view_preparation_path = (
        QUERY_SSL_TEXT_ENCODER_SRC / "query_ssl" / "view_preparation.py"
    )
    source = view_preparation_path.read_text(encoding="utf-8")
    forbidden_snippets = (
        'view_builder_name == "usb_multiview"',
        'view_builder_name == "usb_weak_strong_pair"',
        'view_builder_name == "usb_weak"',
        'augmenter_type == "precomputed_usb_candidates"',
        'augmenter_type != "nllb_backtranslation"',
        "rows_have_usb_multiview_candidates",
        "validate_usb_multiview_candidate_rows",
        "validate_usb_weak_rows",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not legacy_script_path.exists(), (
        "Query SSL unlabeled row augmentation/view preparation core는 "
        "methods/adaptation/query_text_views가 소유한다. scripts는 Hydra cfg와 "
        "runtime callable 주입만 맡긴다."
    )
    assert not violations, (
        "query_ssl_text_encoder script adapter는 USB view builder나 "
        "augmentation source 정책을 직접 분기하지 않는다.\n"
        f"violations={violations}"
    )


def test_query_ssl_text_encoder_runner_stays_descriptor_capability_driven() -> None:
    runner_source = (
        QUERY_SSL_TEXT_ENCODER_SRC / "runners" / "consistency.py"
    ).read_text(encoding="utf-8")
    forbidden_snippets = (
        'algorithm_name == "comatch"',
        'algorithm_name == "simmatch"',
        'algorithm_name == "softmatch"',
        'algorithm_name == "mixmatch"',
        'algorithm_name == "remixmatch"',
        'algorithm_name == "refixmatch"',
        'algorithm_name == "vat"',
        'view_builder_name == "usb_weak_strong_pair"',
        'if "comatch"',
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in runner_source]

    assert not violations, (
        "query_ssl_text_encoder consistency runner는 method 이름이나 concrete view "
        "이름으로 분기하지 않는다. Descriptor required_views/runtime_requirements와 "
        "methods-owned view builder를 통해 capability를 해석해야 한다.\n"
        f"violations={violations}"
    )


def test_common_ssl_hooks_do_not_own_method_specific_hooks() -> None:
    method_fragments = (
        "AdaMatch",
        "CoMatch",
        "Dash",
        "FixMatch",
        "ReFixMatch",
        "ReMixMatch",
        "FlexMatch",
        "FreeMatch",
        "MixMatch",
        "SimMatch",
        "SoftMatch",
        "adamatch",
        "comatch",
        "dash",
        "fixmatch",
        "refixmatch",
        "remixmatch",
        "flexmatch",
        "freematch",
        "mixmatch",
        "simmatch",
        "softmatch",
    )
    violations = [
        f"{_relative_repo_path(path)}: {fragment}"
        for path in _iter_python_files(METHODS_SSL_SRC / "hooks")
        for fragment in method_fragments
        if fragment in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "methods/ssl/hooks는 여러 SSL algorithm에서 안정적으로 공유되는 mechanism만 "
        "소유한다. 단일 method 이름이 붙은 hook/state 조합은 "
        "methods/ssl/algorithms/<method>/ 아래에 둔다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_central_ssl_consistency_entrypoint_imports_runner_directly() -> None:
    entrypoint_path = (
        SCRIPTS_SRC
        / "experiments"
        / "central"
        / "ssl_control"
        / "run_peft_ssl_control.py"
    )
    entrypoint_config = (
        CONF_SRC
        / "entrypoints"
        / "central"
        / "ssl_control"
        / "run_peft_ssl_control.yaml"
    )
    source = entrypoint_path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "run_central_ssl_mode",
        "load_configured_callable",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert "run_query_ssl_peft_baseline" in source
    assert "group_by_query_ssl_method: true" in entrypoint_config.read_text(
        encoding="utf-8"
    )
    assert not violations, (
        "central SSL consistency entrypoint는 explicit workflow 진입점이므로 "
        "generic mode router를 통하지 않는다.\n"
        f"violations={violations}"
    )


def test_dataset_pipeline_download_sources_are_config_declared() -> None:
    source = (
        SCRIPTS_SRC / "workflows" / "datasets" / "run_dataset_pipeline.py"
    ).read_text(encoding="utf-8")
    forbidden_snippets = (
        'source_kind == "huggingface"',
        'source_kind == "kaggle"',
        '"unsupported kind"',
        "download_huggingface_dataset_to_csv(",
        "download_kaggle_dataset_file_to_csv(",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "dataset pipeline runner는 source provider 이름을 직접 분기하지 않는다. "
        "dataset asset YAML의 sources.<name>.download.callable_path가 download "
        "adapter를 선언하고 runner는 configured callable만 실행한다.\n"
        f"violations={violations}"
    )


def test_query_peft_artifact_paths_do_not_branch_on_ssl_input_mode_names() -> None:
    path = QUERY_SSL_TEXT_ENCODER_IO_SRC / "artifact_paths.py"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        'ssl_input_mode != "consistency"',
        'ssl_input_mode == "consistency"',
        'ssl_input_mode == "pseudo_label_replay"',
        "central_ssl_runner",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "central SSL output grouping 규칙은 "
        "entrypoint top-level flag가 소유한다. artifact_paths.py는 "
        "group_by_query_ssl_method만 읽는다.\n"
        f"violations={violations}"
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
        "FL local update profile 실행값은 "
        "conf/strategy_axes/ssl_objective/local_update_profile Hydra YAML이 소유한다. "
        "Python에는 profile별 objective mapping catalog를 다시 만들지 않는다."
    )


def test_fl_local_update_profiles_do_not_keep_lora_classifier_leaf() -> None:
    profile_root = CONF_SRC / "strategy_axes" / "ssl_objective" / "local_update_profile"
    forbidden_path = profile_root / "lora_pseudo_label_v1.yaml"
    assert not forbidden_path.exists(), (
        "active FL local update profile leaf는 peft_classifier_update_v1을 사용한다. "
        "lora_pseudo_label_v1은 old-run artifact/report reader compatibility "
        "표면으로만 남기고 Hydra 실행 profile로 되살리지 않는다."
    )


def test_legacy_fl_strategy_axis_group_is_removed() -> None:
    legacy_root = CONF_SRC / "strategy_axes" / "fl"

    assert not legacy_root.exists(), (
        "active FL strategy axes는 strategy_axes/fl_topology, "
        "strategy_axes/fssl_method, strategy_axes/ssl_objective/local_update_profile로 "
        "나뉜다. legacy strategy_axes/fl group은 README/marker만 남겨도 새 축 위치를 "
        "흐리므로 되살리지 않는다."
    )


def test_fl_client_split_preset_is_not_strategy_axis() -> None:
    forbidden_path = CONF_SRC / "strategy_axes" / "fl_topology" / "materialized_split"
    expected_path = CONF_SRC / "execution_context" / "fl_client_split"

    assert not forbidden_path.exists(), (
        "materialized FL client split preset은 method/topology strategy axis가 아니라 "
        "실행 데이터 artifact 선택이다. execution_context/fl_client_split 아래에 둔다."
    )
    assert expected_path.exists(), (
        "FL client split preset 선택 group은 execution_context/fl_client_split에 둔다."
    )


def test_central_ssl_input_mode_strategy_axis_group_is_removed() -> None:
    legacy_root = CONF_SRC / "strategy_axes" / "ssl_objective" / "input_mode"

    assert not legacy_root.exists(), (
        "central SSL은 explicit consistency entrypoint가 workflow를 고르고, "
        "input_mode를 public strategy axis로 다시 열지 않는다. "
        "pseudo-label replay는 별도 workflow로만 두고 teacher bootstrap helper는 "
        "scripts에 되살리지 않는다."
    )


def test_central_ssl_entrypoint_does_not_compose_input_mode_strategy_axis() -> None:
    path = (
        CONF_SRC
        / "entrypoints"
        / "central"
        / "ssl_control"
        / "run_peft_ssl_control.yaml"
    )
    source = path.read_text(encoding="utf-8")

    assert "strategy_axes/ssl_objective/input_mode" not in source, (
        "central SSL root entrypoint는 consistency method와 scaffold 조합만 소유한다. "
        "input_mode는 workflow/helper 내부 값으로 격리한다."
    )


def test_query_peft_support_does_not_emit_ssl_input_mode_manifest_field() -> None:
    search_roots = (
        QUERY_SSL_TEXT_ENCODER_SRC,
        SCRIPTS_SRC / "experiments" / "central" / "ssl_control",
        CONF_SRC / "entrypoints" / "central" / "ssl_control",
    )
    violations = [
        _relative_repo_path(path)
        for root in search_roots
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".yaml", ".md"}
        and "ssl_input_mode" in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "`ssl_input_mode`는 제거된 input_mode strategy axis의 legacy manifest "
        "표식이다. "
        "workflow-specific metadata가 필요하면 active runner의 이름 있는 payload로 "
        "남긴다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_central_ssl_teacher_provider_strategy_axis_group_is_removed() -> None:
    legacy_root = CONF_SRC / "strategy_axes" / "ssl_objective" / "teacher_provider"

    assert not legacy_root.exists(), (
        "teacher source는 독립 teacher_provider strategy axis가 아니다. "
        "중앙 SSL에서는 method hook/recipe가 teacher source 의미를 소유한다."
    )


def test_central_ssl_pseudo_label_selection_strategy_axis_group_is_removed() -> None:
    legacy_root = (
        CONF_SRC / "strategy_axes" / "ssl_objective" / "pseudo_label_selection"
    )

    assert not legacy_root.exists(), (
        "pseudo_label_selection은 중앙 SSL public strategy axis가 아니다. "
        "selection hook은 methods/ssl/hooks가 소유하고, recipe 기본값이나 "
        "ablation metadata로만 연결한다."
    )


def test_query_peft_offline_pseudo_label_replay_workflow_is_removed() -> None:
    removed_paths = (
        QUERY_SSL_TEXT_ENCODER_SRC / "runners" / "pseudo_label.py",
        QUERY_SSL_TEXT_ENCODER_SRC / "runners" / "pseudo_label_inputs.py",
        METHODS_SSL_SRC / "pseudo_label_replay.py",
        METHODS_SSL_SRC / "teacher_pseudo_label.py",
        QUERY_SSL_TEXT_ENCODER_IO_SRC / "teacher_pseudo_label_artifact_writer.py",
    )
    existing_paths = [path for path in removed_paths if path.exists()]

    assert not existing_paths, (
        "offline pseudo-label replay/self-training workflow는 중앙 online SSL "
        "canonical surface가 아니다.\n"
        f"{chr(10).join(f'- {_relative_repo_path(path)}' for path in existing_paths)}"
    )


def test_query_peft_agent_local_query_adaptation_export_bridge_is_removed() -> None:
    removed_paths = (
        QUERY_SSL_TEXT_ENCODER_SRC / "runners" / "query_adaptation.py",
        QUERY_SSL_TEXT_ENCODER_IO_SRC / "query_adaptation.py",
        QUERY_SSL_TEXT_ENCODER_IO_SRC / "query_adaptation_multiview.py",
    )
    existing_paths = [path for path in removed_paths if path.exists()]

    assert not existing_paths, (
        "agent-local query adaptation export bridge는 중앙 지도/중앙 SSL/FSSL "
        "canonical experiment surface가 아니다. "
        "agent-local adaptation dataset runtime은 agent owner에 남기고, 중앙 "
        "실험은 supervised/consistency runner와 Hydra initial_checkpoint로 연결한다.\n"
        f"{chr(10).join(f'- {_relative_repo_path(path)}' for path in existing_paths)}"
    )


def test_query_peft_teacher_bootstrap_compatibility_tree_is_removed() -> None:
    legacy_root = QUERY_SSL_TEXT_ENCODER_SRC / "compatibility" / "teacher_bootstrap"

    assert not legacy_root.exists(), (
        "teacher_bootstrap은 scripts owner가 아닌 fixed-classifier compatibility "
        "debt였다. "
        "새 teacher source가 필요하면 methods/ssl hook 또는 method recipe로 추가한다.\n"
        f"legacy path={_relative_repo_path(legacy_root)}"
    )


def test_fl_local_ssl_policy_does_not_expose_method_local_fedmatch_leaf() -> None:
    forbidden_path = (
        CONF_SRC
        / "strategy_axes"
        / "ssl_objective"
        / "local_ssl_policy"
        / "fedmatch_agreement.yaml"
    )

    assert not forbidden_path.exists(), (
        "fedmatch_agreement는 FedMatch method-local objective다. generic "
        "local_ssl_policy Hydra leaf로 선택하지 말고 FedMatch descriptor와 "
        "scenario default에서 파생한다."
    )


def test_fl_local_ssl_policy_does_not_point_to_local_update_profile() -> None:
    policy_root = CONF_SRC / "strategy_axes" / "ssl_objective" / "local_ssl_policy"
    violations: list[Path] = []
    for path in policy_root.glob("*.yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if payload.get("parameter_source") == "local_update_profile":
            violations.append(path)

    assert not violations, (
        "local_ssl_policy는 SSL method 파라미터 출처를 뜻한다. "
        "local_update_profile은 update backend/example/privacy recipe만 소유하므로 "
        "pseudo-label/selection/scoring 파라미터 출처로 쓰면 안 된다.\n"
        f"{chr(10).join(f'- {_relative_repo_path(path)}' for path in violations)}"
    )


def test_fl_server_update_policy_does_not_expose_method_local_fedmatch_leaf() -> None:
    forbidden_path = (
        CONF_SRC
        / "strategy_axes"
        / "fl_topology"
        / "server_update"
        / "fedmatch_partitioned.yaml"
    )

    assert not forbidden_path.exists(), (
        "fedmatch_partitioned는 FedMatch method-local server update policy다. "
        "generic server_update_policy Hydra leaf로 선택하지 말고 "
        "FedMatch descriptor와 scenario default에서 파생한다."
    )


def test_generic_fl_ssl_compatibility_does_not_own_fedmatch_policy_rules() -> None:
    source = (METHODS_FEDERATED_SSL_SRC / "compatibility.py").read_text(
        encoding="utf-8"
    )
    forbidden_snippets = (
        "SERVER_UPDATE_FEDMATCH_PARTITIONED",
        "LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT",
        "LOCAL_SSL_POLICY_FIXMATCH",
        "fedmatch_partitioned",
        "fedmatch_agreement",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "methods/federated_ssl/compatibility.py는 capability 공통 규칙만 소유한다. "
        "FedMatch partitioned server update와 agreement/fixmatch 허용 조합은 "
        "methods/federated_ssl/fedmatch/compatibility.py가 소유한다.\n"
        f"violations={violations}"
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


def test_fl_entrypoint_does_not_own_payload_payload_adapter_alias() -> None:
    path = CONF_SRC / "entrypoints" / "fl_ssl" / "run_federated_simulation.yaml"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "payload_adapter_name:",
        "payload_adapter_kind:",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "FL root entrypoint는 실행 update family와 aggregation backend만 조합한다. "
        "v1 payload adapter kind compatibility alias는 "
        "strategy_axes/model_architecture/update_family leaf가 소유한다.\n"
        f"violations={violations}"
    )


def test_fl_simulation_runtime_model_does_not_embed_lora_classifier_scope() -> None:
    checked_paths = (
        SCRIPTS_SRC / "experiments" / "fl_ssl" / "federated_simulation" / "models.py",
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "config_request.py",
        METHODS_SRC
        / "adaptation"
        / "peft_text_encoder"
        / "simulation_runtime"
        / "round_runtime.py",
        METHODS_SRC / "adaptation" / "peft_text_encoder" / "update_family_runtime.py",
        METHODS_SRC / "adaptation" / "peft_text_encoder" / "resource_cache.py",
    )
    forbidden_snippets = (
        'round_runtime_payloads.get("lora_classifier")',
        'round_runtime_mapping.get("lora_classifier")',
        "lora_classifier: FederatedPeftEncoderRuntimeConfig",
        "round_runtime_config.lora_classifier",
        "or round_runtime_config.lora_classifier",
        'payload_adapter_name == "lora_classifier"',
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
        / "generic_client_runtime_bridge.py",
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "adapters"
        / "runtime_compatibility.py",
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "io"
        / "protocol_payload.py",
        CONF_SRC / "entrypoints" / "fl_ssl" / "run_federated_simulation.yaml",
        SCRIPTS_SRC / "experiments" / "fl_ssl" / "federated_simulation" / "README.md",
        METHODS_SRC / "adaptation" / "peft_text_encoder" / "update_family_runtime.py",
        METHODS_SRC
        / "adaptation"
        / "peft_text_encoder"
        / "simulation_runtime"
        / "round_runtime.py",
    )
    forbidden_snippets = (
        "peft_classifier: FederatedPeftEncoderRuntimeConfig",
        'round_runtime_payloads.get("peft_classifier")',
        'round_runtime_mapping.get("peft_classifier")',
        "round_runtime_mapping.get(PEFT_CLASSIFIER_ADAPTER_KIND)",
        "round_runtime.peft_classifier",
        "round_runtime.lora_classifier",
        "runtime_payload_for_payload_adapter",
        "classifier_head_bootstrap_logit_scale",
        "payload_adapter_name=str(cfg.round_runtime.payload_adapter_name)",
        "payload_adapter_name: str | None",
        "def payload_adapter_name(",
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


def test_fl_run_layout_does_not_own_labeled_exposure_policy_slug_map() -> None:
    path = SCRIPTS_SRC / "experiments" / "fl_ssl" / "support" / "layout.py"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "def _compact_labeled_exposure_slug(",
        "policy_name == LABELED_EXPOSURE_SHARED_CLIENT_SEED",
        "policy_name == LABELED_EXPOSURE_SERVER_ONLY_SEED",
        "policy_name == LABELED_EXPOSURE_CLIENT_LOCAL_SPLIT",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "labeled exposure policy의 path/report용 compact slug는 "
        "methods/federated/client_split.py가 소유한다. run_layout은 artifact path "
        "조립만 맡는다.\n"
        f"violations={violations}"
    )


def test_fl_simulation_does_not_own_labeled_exposure_row_policy() -> None:
    checked_paths = (
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "data_source_request.py",
        SCRIPTS_SRC / "experiments" / "fl_ssl" / "materialize_fl_client_split.py",
    )
    forbidden_snippets = (
        "labeled_exposure_policy.name == LABELED_EXPOSURE_SERVER_ONLY_SEED",
        "labeled_exposure_policy.name == LABELED_EXPOSURE_CLIENT_LOCAL_SPLIT",
        "resolved_labeled_exposure_policy.name == LABELED_EXPOSURE_SERVER_ONLY_SEED",
        "resolved_labeled_exposure_policy.name == LABELED_EXPOSURE_CLIENT_LOCAL_SPLIT",
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for path in checked_paths
        for snippet in forbidden_snippets
        if path.exists() and snippet in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "labeled exposure policy의 client/bootstrap row 노출 의미는 "
        "methods/federated/client_split.py가 소유한다. FL simulation/materialization "
        "adapter는 rows를 읽고 helper에 넘긴다.\n"
        f"violations={violations}"
    )


def test_peft_runtime_bridges_use_update_family_for_support_checks() -> None:
    checked_paths = (
        METHODS_SRC / "adaptation" / "peft_text_encoder" / "update_family_runtime.py",
        SCRIPTS_RUNTIME_ADAPTER_SRC
        / "federated_agent"
        / "generic_client_runtime_bridge.py",
        SCRIPTS_RUNTIME_ADAPTER_SRC
        / "federated_server"
        / "generic_server_runtime_bridge.py",
    )
    forbidden_snippets = (
        "is_peft_encoder_payload_adapter",
        "round_runtime_config.payload_adapter_name",
        "payload_adapter_name:",
    )
    violations: list[tuple[Path, str]] = []
    for path in checked_paths:
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in source:
                violations.append((_relative_repo_path(path), snippet))

    assert not violations, (
        "PEFT runtime bridge의 지원 여부와 payload 선택은 update_family_name 기준이다. "
        "payload_adapter_name은 shared contract/aggregation compatibility 표면에만 "
        "남긴다.\n"
        f"{chr(10).join(f'- {path}: {snippet}' for path, snippet in violations)}"
    )


def test_active_round_runtime_configs_do_not_accept_payload_adapter_alias() -> None:
    checked_paths = (
        SCRIPTS_SRC / "experiments" / "fl_ssl" / "federated_simulation" / "models.py",
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "config_request.py",
        REPO_ROOT
        / "main_server"
        / "src"
        / "services"
        / "federation"
        / "rounds"
        / "runtime"
        / "config.py",
    )
    forbidden_snippets = (
        "payload_adapter_name: str | None",
        "def payload_adapter_name(",
        "LEGACY_ROUND_ADAPTER_FAMILY_ENV",
        "TRACEMIND_ROUND_ADAPTER_FAMILY",
        'round_runtime, "payload_adapter_name"',
        "provide legacy round_runtime.payload_adapter_name",
        "payload_adapter_kind and legacy payload_adapter_name",
    )
    violations: list[tuple[Path, str]] = []
    for path in checked_paths:
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            if snippet in source:
                violations.append((_relative_repo_path(path), snippet))

    assert not violations, (
        "active FL runtime config는 payload_adapter_kind만 받는다. "
        "payload_adapter_name은 새 실행 config, report/result reader alias로 "
        "되살리지 않는다.\n"
        f"{chr(10).join(f'- {path}: {snippet}' for path, snippet in violations)}"
    )


def test_fl_report_protocol_records_payload_adapter_kind() -> None:
    required_by_path = (
        (
            SCRIPTS_SRC
            / "experiments"
            / "fl_ssl"
            / "federated_simulation"
            / "io"
            / "protocol_payload.py",
            ('"payload_adapter_kind": round_runtime_config.payload_adapter_kind',),
        ),
        (
            SCRIPTS_SRC / "workflows" / "result_index" / "fl_ssl_report_loader.py",
            ('round_runtime.get("payload_adapter_kind")',),
        ),
    )
    missing = [
        f"{_relative_repo_path(path)}: {snippet}"
        for path, snippets in required_by_path
        for snippet in snippets
        if snippet not in path.read_text(encoding="utf-8")
    ]

    assert not missing, (
        "새 FL report protocol은 update_family_name과 payload_adapter_kind를 "
        "canonical field로 기록한다. payload_adapter_name fallback은 result reader에도 "
        "되살리지 않는다.\n"
        f"{chr(10).join(f'- {item}' for item in missing)}"
    )


def test_result_index_uses_payload_adapter_kind_as_canonical_field() -> None:
    required_by_path = (
        (
            SCRIPTS_SRC / "workflows" / "result_index" / "models.py",
            ("payload_adapter_kind: str | None",),
        ),
        (
            SCRIPTS_SRC / "workflows" / "result_index" / "schema.py",
            ("payload_adapter_kind text",),
        ),
        (
            SCRIPTS_SRC / "workflows" / "result_index" / "dashboard_export.py",
            ('"payload_adapter_kinds"',),
        ),
        (
            REPO_ROOT
            / "apps"
            / "experiment_dashboard"
            / "src"
            / "features"
            / "fl_ssl"
            / "logic"
            / "labels.js",
            ("row.payload_adapter_kind", "payload_adapter_kind"),
        ),
    )
    forbidden_by_path = (
        (
            SCRIPTS_SRC / "workflows" / "result_index" / "models.py",
            ("payload_adapter_name: str | None",),
        ),
        (
            SCRIPTS_SRC / "workflows" / "result_index" / "schema.py",
            ("payload_adapter_name text",),
        ),
        (
            SCRIPTS_SRC / "workflows" / "result_index" / "dashboard_export.py",
            ('"adapter_families"',),
        ),
    )
    missing = [
        f"{_relative_repo_path(path)}: {snippet}"
        for path, snippets in required_by_path
        for snippet in snippets
        if snippet not in path.read_text(encoding="utf-8")
    ]
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for path, snippets in forbidden_by_path
        for snippet in snippets
        if path.exists() and snippet in path.read_text(encoding="utf-8")
    ]

    assert not missing and not violations, (
        "result-index와 dashboard의 canonical 실행 표면은 payload_adapter_kind다. "
        "payload_adapter_name은 old report/old DB fallback reader로도 해석하지 "
        "않는다.\n"
        f"missing:\n{chr(10).join(f'- {item}' for item in missing)}\n"
        f"violations:\n{chr(10).join(f'- {item}' for item in violations)}"
    )


def test_federated_ssl_active_docs_use_update_family_terms() -> None:
    checked_paths = (
        METHODS_FEDERATED_SSL_SRC / "README.md",
        METHODS_FEDERATED_SSL_SRC / "fedmatch" / "README.md",
        METHODS_FEDERATED_SSL_SRC / "fedmatch" / "partitioning.py",
    )
    forbidden_snippets = (
        "LoRA-classifier",
        "lora_classifier_training.py",
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for path in checked_paths
        for snippet in forbidden_snippets
        if path.exists() and snippet in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "active federated_ssl method 문서는 PEFT encoder/update family 용어를 "
        "사용한다. v1 LoRA-classifier 이름은 legacy contract/audit이나 실제 "
        "compatibility adapter 표면에만 남긴다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_federated_ssl_runtime_pairs_are_update_family_oriented() -> None:
    checked_roots = (
        METHODS_FEDERATED_SSL_SRC,
        TEST_FIXTURES_SRC,
    )
    violations: list[tuple[Path, int]] = []
    for root in checked_roots:
        for path in _iter_python_files(root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Name):
                    continue
                if node.func.id != "FederatedSslRuntimePair":
                    continue
                if any(
                    keyword.arg == "payload_adapter_name" for keyword in node.keywords
                ):
                    violations.append((_relative_repo_path(path), node.lineno))

    assert not violations, (
        "FL SSL method recipe의 runtime pair는 trainable-state/update-family "
        "조합을 표현한다. payload_adapter_name은 shared payload/aggregation "
        "compatibility 표면에만 남긴다.\n"
        f"{chr(10).join(f'- {path}:{line}' for path, line in violations)}"
    )


def test_fl_simulation_server_aggregate_namespace_uses_update_family() -> None:
    checked_paths = (
        SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_server" / "runtime.py",
        SCRIPTS_RUNTIME_ADAPTER_SRC / "federated_server" / "aggregation_artifacts.py",
        SCRIPTS_RUNTIME_ADAPTER_SRC
        / "federated_server"
        / "generic_server_runtime_bridge.py",
        REPO_ROOT / "tests" / "unit" / "test_run_federated_simulation.py",
    )
    forbidden_snippets = (
        "server-aggregate://{payload_adapter_name}",
        "server-aggregate://peft_classifier",
        '/ "peft_classifier"\n        / "sim_rev_',
        "payload_adapter_name=str(active.adapter_state.adapter_kind)",
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


def test_fl_simulation_does_not_own_payload_adapter_compatibility_rule() -> None:
    path = (
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "simulation.py"
    )
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "def _require_local_adapter_matches_round_runtime(",
        "local_update_profile and round_runtime",
        "round_payload_adapter={request.round_runtime_config.payload_adapter_name}",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "FL SSL payload-adapter compatibility rule과 error message는 "
        "methods/federated_ssl/compatibility.py가 소유한다. simulation runner는 "
        "bootstrap에서 methods-owned validator만 호출한다.\n"
        f"violations={violations}"
    )


def test_fl_simulation_unit_tests_use_active_peft_payload_surface() -> None:
    source = (
        REPO_ROOT / "tests" / "unit" / "test_run_federated_simulation.py"
    ).read_text(encoding="utf-8")
    forbidden_snippets = (
        "make_lora_classifier_delta_payload",
        "LORA_CLASSIFIER_UPDATE_PAYLOAD_FORMAT",
        'payload_kind="lora_classifier_materialized_state.v1"',
        'payload_adapter_name="diagonal_scale"',
        'update_family_name="diagonal_scale"',
        'training_backend_name="diagonal_scale_heuristic"',
        'privacy_guard_name="diagonal_scale_clip_only"',
        'scorer_backend_name="diagonal_scale_logits"',
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "FL simulation unit fixture는 active PEFT text encoder payload surface를 "
        "검증한다. v1 lora_classifier/diagonal_scale payload 직접 생성은 shared "
        "contract compatibility 테스트로 격리한다.\n"
        f"{chr(10).join(f'- {snippet}' for snippet in violations)}"
    )


def test_scripts_runtime_bridges_use_peft_config_type_names() -> None:
    checked_paths = (
        SCRIPTS_RUNTIME_ADAPTER_SRC
        / "federated_agent"
        / "generic_client_runtime_bridge.py",
        METHODS_SRC
        / "adaptation"
        / "peft_text_encoder"
        / "simulation_runtime"
        / "round_runtime.py",
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
    path = PEFT_TEXT_ENCODER_SRC / "config.py"
    source = path.read_text(encoding="utf-8")
    required_snippets = (
        "class PeftEncoderTrainingBackendConfig:",
        'artifact_ref_prefix: str = "agent-local://peft_classifier"',
        "payload_adapter_kind: str = PEFT_ENCODER_PAYLOAD_ADAPTER_KIND",
        "PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL",
        "PEFT_ENCODER_DELTA_FORMAT_INLINE",
        "PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED",
    )
    missing = [snippet for snippet in required_snippets if snippet not in source]
    forbidden_snippets = (
        "PeftClassifierTrainingBackendConfig = LoraClassifierTrainingBackendConfig",
        "PeftEncoderTrainingBackendConfig = LoraClassifierTrainingBackendConfig",
        "class LoraClassifierTrainingBackendConfig",
        "def build_legacy_lora_classifier_training_backend_config(",
        "def build_lora_classifier_training_backend_config(",
        'artifact_ref_prefix: str = "agent-local://lora_classifier"',
        "payload_adapter_kind: str = LORA_CLASSIFIER_PAYLOAD_ADAPTER_KIND",
        "LORA_CLASSIFIER_DELTA_FORMAT_AGENT_LOCAL",
        "LORA_CLASSIFIER_DELTA_FORMAT_INLINE",
        "LORA_CLASSIFIER_DELTA_FORMAT_SERVER_UPLOADED",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not missing and not violations, (
        "PEFT encoder config가 canonical class/default를 소유하고, "
        "active training config에서 v1 lora_classifier payload producer를 "
        "다시 열지 않는다.\n"
        f"missing={missing}\nviolations={violations}"
    )


def test_peft_training_backend_does_not_register_legacy_lora_factories() -> None:
    path = PEFT_TEXT_ENCODER_SRC / "training_backend.py"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "def from_objective_config(",
        "def from_legacy_lora_objective_config(",
        "def build_legacy_lora_classifier_training_backend(",
        "def build_lora_classifier_training_backend(",
        "def build_peft_classifier_training_backend(",
        "PEFT_CLASSIFIER_TRAINING_BACKEND_CATALOG_ENTRY",
        '"lora_classifier_trainer"',
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "PEFT training backend는 active v2 peft_classifier_trainer만 등록한다. "
        "v1 lora_classifier_trainer producer alias를 다시 만들지 않는다.\n"
        f"violations={violations}"
    )


def test_local_update_registry_does_not_embed_peft_backend_override() -> None:
    path = METHODS_SRC / "adaptation" / "local_update_registry.py"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "_TRAINING_BACKEND_MODULE_OVERRIDES",
        '"methods.adaptation.peft_text_encoder.training_backend"',
        '"peft_classifier_trainer":',
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "local update registry는 concrete backend module override를 직접 소유하지 "
        "않는다. 먼저 name convention을 시도하고, 필요하면 adaptation package scan으로 "
        "구현 옆 decorator 등록을 로드한다.\n"
        f"{chr(10).join(f'- {snippet}' for snippet in violations)}"
    )


def test_payload_adapter_module_resolver_does_not_embed_concrete_family_map() -> None:
    removed_path = METHODS_SRC / "adaptation" / "payload_adapter_modules.py"
    path = METHODS_SRC / "adaptation" / "implementation_modules.py"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        '"classifier_head":',
        '"lora_classifier":',
        '"peft_classifier":',
        '"methods.adaptation.peft_text_encoder"',
        '"methods.classification.linear_head"',
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not removed_path.exists(), (
        "payload_adapter_modules.py 이름은 legacy payload-adapter 용어를 되살린다. "
        "payload adapter kind -> implementation owner 해석은 "
        "implementation_modules.py가 소유한다."
    )
    assert not violations, (
        "payload adapter module resolver는 concrete alias table을 소유하지 "
        "않는다. alias 선언은 구현 owner 옆 payload_adapter_module manifest에 둔다.\n"
        f"{chr(10).join(f'- {snippet}' for snippet in violations)}"
    )


def test_peft_method_modules_use_canonical_config_type_name() -> None:
    checked_roots = (
        PEFT_TEXT_ENCODER_SRC,
        METHODS_FEDERATED_SSL_SRC / "fedmatch",
    )
    violations = [
        _relative_repo_path(path)
        for root in checked_roots
        for path in _iter_python_files(root)
        if "LoraClassifierTrainingBackendConfig" in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "PEFT text-encoder/head/FedMatch active method modules는 canonical "
        "PeftEncoderTrainingBackendConfig type 이름을 사용한다. v1 "
        "LoraClassifierTrainingBackendConfig subclass/builder를 다시 만들지 "
        "않는다.\n"
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
        for root in (PEFT_TEXT_ENCODER_SRC, METHODS_FEDERATED_SSL_SRC / "fedmatch")
        for path in _iter_python_files(root)
        for snippet in forbidden_snippets
        if path.exists() and snippet in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "PEFT text-encoder/head/FedMatch active method modules는 model/runtime type도 "
        "PeftTextEncoderWithLinearHead와 PeftEncoderModelRuntimeConfig를 쓴다. "
        "LoRA는 PEFT adapter mechanism 이름으로만 남긴다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_partitioned_peft_execution_primitive_uses_adapter_linear_head_names() -> None:
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
        PEFT_TEXT_ENCODER_SRC / "federated_ssl" / "partitioned" / "training_loop.py",
        PEFT_TEXT_ENCODER_SRC / "federated_ssl" / "partitioned_objective_training.py",
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
        if path.exists() and snippet in path.read_text(encoding="utf-8")
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


def test_scripts_use_query_ssl_text_encoder_runtime_support_package_path() -> None:
    legacy_root = SCRIPTS_SRC / "experiments" / "query_lora_ssl"
    legacy_peft_support_root = SCRIPTS_SRC / "support" / "query_ssl_peft"
    checked_roots = (SCRIPTS_SRC, REPO_ROOT / "tests")
    forbidden_snippets = (
        "scripts.experiments." + "query_lora_ssl",
        "scripts/experiments/" + "query_lora_ssl",
        "scripts.support." + "query_ssl_peft",
        "scripts/support/" + "query_ssl_peft",
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for root in checked_roots
        for path in _iter_python_files(root)
        if path != Path(__file__).resolve()
        for snippet in forbidden_snippets
        if path.exists() and snippet in path.read_text(encoding="utf-8")
    ]

    assert (
        QUERY_SSL_TEXT_ENCODER_SRC.is_dir()
        and not legacy_root.exists()
        and not legacy_peft_support_root.exists()
        and not violations
    ), (
        "중앙 Query SSL runtime support package 경로는 text encoder scaffold 기준인 "
        "query_ssl_text_encoder를 사용한다. LoRA/PEFT는 adapter mechanism, "
        "entrypoint 이름, artifact/contract 이름으로만 남기고 scripts support "
        "package boundary 이름으로 재도입하지 않는다.\n"
        f"legacy_exists={legacy_root.exists()}\n"
        f"legacy_peft_support_exists={legacy_peft_support_root.exists()}\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_central_ssl_entrypoints_use_control_names() -> None:
    expected_paths = (
        SCRIPTS_SRC
        / "experiments"
        / "central"
        / "ssl_control"
        / "run_peft_ssl_control.py",
        SCRIPTS_SRC
        / "experiments"
        / "central"
        / "ssl_control"
        / "run_peft_supervised_control.py",
        SCRIPTS_SRC
        / "experiments"
        / "central"
        / "ssl_control"
        / "run_full_text_encoder_supervised_control.py",
        CONF_SRC
        / "entrypoints"
        / "central"
        / "ssl_control"
        / "run_peft_ssl_control.yaml",
        CONF_SRC
        / "entrypoints"
        / "central"
        / "ssl_control"
        / "run_peft_supervised_control.yaml",
        CONF_SRC
        / "entrypoints"
        / "central"
        / "ssl_control"
        / "run_full_text_encoder_supervised_control.yaml",
    )
    legacy_paths = (
        SCRIPTS_SRC
        / "experiments"
        / "central"
        / "ssl_control"
        / "train_lora_ssl_classifier.py",
        SCRIPTS_SRC
        / "experiments"
        / "central"
        / "ssl_control"
        / "train_lora_supervised_classifier.py",
        CONF_SRC
        / "entrypoints"
        / "central"
        / "ssl_control"
        / "train_lora_ssl_classifier.yaml",
        CONF_SRC
        / "entrypoints"
        / "central"
        / "ssl_control"
        / "train_lora_supervised_classifier.yaml",
        SCRIPTS_SRC
        / "experiments"
        / "central"
        / "ssl_control"
        / "train_peft_ssl_classifier.py",
        SCRIPTS_SRC
        / "experiments"
        / "central"
        / "ssl_control"
        / "train_peft_supervised_classifier.py",
        CONF_SRC
        / "entrypoints"
        / "central"
        / "ssl_control"
        / "train_peft_ssl_classifier.yaml",
        CONF_SRC
        / "entrypoints"
        / "central"
        / "ssl_control"
        / "train_peft_supervised_classifier.yaml",
    )
    checked_paths = (
        SCRIPTS_SRC / "README.md",
        SCRIPTS_SRC / "experiments" / "README.md",
        SCRIPTS_SRC / "experiments" / "central" / "ssl_control" / "README.md",
        SCRIPTS_SRC / "support" / "query_ssl_text_encoder" / "README.md",
    )
    forbidden_snippets = (
        "train_lora_ssl_classifier",
        "train_lora_supervised_classifier",
        "train_peft_ssl_classifier",
        "train_peft_supervised_classifier",
        "PEFT encoder classifier",
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
        if path.exists() and snippet in path.read_text(encoding="utf-8")
    ]

    assert not (missing_paths or legacy_existing_paths or violations), (
        "중앙 SSL 실행 entrypoint/config/README는 classifier 전용 이름이 아니라 "
        "PEFT text encoder control 실행 표면으로 드러낸다. old-run reader만 과거 "
        "entrypoint/output-dir 이름을 해석할 수 있다.\n"
        f"missing={missing_paths}\n"
        f"legacy_existing={legacy_existing_paths}\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_central_peft_entrypoints_do_not_write_lora_named_artifact_roots() -> None:
    checked_paths = (
        CONF_SRC
        / "entrypoints"
        / "central"
        / "ssl_control"
        / "run_peft_ssl_control.yaml",
        CONF_SRC
        / "entrypoints"
        / "central"
        / "ssl_control"
        / "run_peft_supervised_control.yaml",
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
        if path.exists() and snippet in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "새 중앙 PEFT 실행 산출물 기본 root는 PEFT/query 의미로 이름 붙인다. "
        "LoRA는 adapter mechanism 또는 old artifact reader compatibility 표면에만 "
        "남긴다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_query_ssl_text_encoder_runtime_support_keeps_surface_names_separated() -> None:
    forbidden_snippets = (
        "query_" + "lora",
        "Query" + "Lora",
        "run_query_ssl_" + "lora",
        "train_query_ssl_" + "lora",
        "Supervised" + "Lora",
        "Lora" + "Labeled",
        "evaluate_lora_run_context",
        "run_fixed_classifier_teacher_lora_student_bootstrap",
        "_run_student_lora_bootstrap",
        "fixed_classifier_lora_bootstrap.v1",
        "query_adapt_lora",
        "lora_bootstrap",
        "lora_clf",
        "SupervisedPeftRunContext",
        "PeftLabeledRunContext",
        "prepare_supervised_peft_run_context",
        "evaluate_supervised_peft_run_context",
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for path in _iter_python_files(QUERY_SSL_TEXT_ENCODER_SRC)
        for snippet in forbidden_snippets
        if path.exists() and snippet in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "query_ssl_text_encoder runtime support 내부 공통 helper/type 이름은 "
        "trainable surface와 PEFT adapter mechanism 이름을 섞지 않는다. "
        "PEFT는 PEFT entrypoint/runner/artifact 이름에만, LoRA는 adapter mechanism이나 "
        "old-run artifact/entrypoint compatibility 표면에만 남긴다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_query_ssl_text_encoder_common_context_does_not_default_to_peft() -> None:
    checked_paths = (
        QUERY_SSL_TEXT_ENCODER_SRC / "text_encoder_run_context.py",
        QUERY_SSL_TEXT_ENCODER_SRC / "query_ssl" / "run_context.py",
        QUERY_SSL_TEXT_ENCODER_SRC / "runners" / "supervised_text_encoder.py",
    )
    forbidden_snippet = "methods.adaptation.peft_text_encoder"
    violations = [
        _relative_repo_path(path)
        for path in checked_paths
        if forbidden_snippet in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "query_ssl_text_encoder 공통 context는 PEFT model builder를 기본값으로 "
        "소유하지 않는다. PEFT runner가 PEFT builder를 주입하고, full/frozen/prototype "
        "runner는 자기 surface builder를 주입해야 한다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_result_index_and_dashboard_use_peft_adapter_fields() -> None:
    checked_paths = (
        SCRIPTS_SRC / "workflows" / "result_index" / "schema.py",
        SCRIPTS_SRC / "workflows" / "result_index" / "models.py",
        SCRIPTS_SRC / "workflows" / "result_index" / "fl_ssl_report_loader.py",
        SCRIPTS_SRC / "workflows" / "result_index" / "dashboard_export.py",
    )
    forbidden_snippets = (
        "LoRA-classifier",
        "central_lora_ssl",
        "central_lora_initial_eval",
        "adapter_family_name",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_bias",
        "lora_target_modules",
        "lora_use_rslora",
        "lora_use_dora",
        "peft_adapter_use_rslora",
        "peft_adapter_use_dora",
        "lora_ranks",
        "lora_alphas",
        "replace(/^lora_/",
        "loraConfigLabel",
        "loraVariantLabel",
        "sameLoraConfig",
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for path in checked_paths
        for snippet in forbidden_snippets
        if path.exists() and snippet in path.read_text(encoding="utf-8")
    ]
    dashboard_paths = tuple(
        path
        for path in (REPO_ROOT / "apps" / "experiment_dashboard" / "src").rglob("*.js")
        if path.name != "normalize_bundle.js"
    )
    violations.extend(
        f"{_relative_repo_path(path)}: {snippet}"
        for path in dashboard_paths
        for snippet in forbidden_snippets
        if snippet in path.read_text(encoding="utf-8")
    )

    assert not violations, (
        "result index와 dashboard 계약은 PEFT adapter field를 사용한다. LoRA는 "
        "adapter mechanism 값이나 old report reader fallback에만 남기고, DB/UI "
        "상위 field 이름으로 고정하지 않는다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_peft_partition_delta_uses_canonical_internal_type_name() -> None:
    checked_roots = (
        PEFT_TEXT_ENCODER_SRC,
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
        "project_adapter_linear_head_delta_bundle_to_lora_partition_delta",
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for root in checked_roots
        for path in (_iter_python_files(root) if root.is_dir() else [root])
        for snippet in forbidden_snippets
        if path.exists() and snippet in path.read_text(encoding="utf-8")
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
        "test_upload_agent_local_peft_update_materializes_server_owned_refs",
    )
    missing = [snippet for snippet in required_snippets if snippet not in source]

    assert not violations and not missing, (
        "federated agent runtime adapter unit test는 active 경로를 PEFT encoder로 "
        "부르고, 새 payload producer는 PEFT classifier 계약을 사용한다.\n"
        f"violations={violations}\nmissing={missing}"
    )


def test_federated_ssl_client_diagnostics_use_method_discovery() -> None:
    source = (METHODS_FEDERATED_SSL_SRC / "diagnostics" / "client.py").read_text(
        encoding="utf-8"
    )
    forbidden_snippets = (
        "_KNOWN_METHOD_DIAGNOSTIC_MODULES",
        '("fedmatch",)',
        "for method_name in _KNOWN",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "method-local client diagnostics는 methods/federated_ssl/<method>/"
        "client_diagnostics.py convention으로 발견한다. 새 FL method 추가 때 "
        "공통 diagnostics/client.py에 method 이름 목록을 누적하지 않는다.\n"
        f"violations={violations}"
    )


def test_round_state_exchange_names_are_contract_owned() -> None:
    base_source = (METHODS_FEDERATED_SSL_SRC / "base.py").read_text(encoding="utf-8")
    execution_plan_source = (METHODS_FEDERATED_SSL_SRC / "execution_plan.py").read_text(
        encoding="utf-8"
    )
    executor_source = (
        MAIN_SERVER_SRC
        / "services"
        / "federation"
        / "rounds"
        / "round_state_exchange"
        / "executor.py"
    ).read_text(encoding="utf-8")

    assert 'ROUND_STATE_EXCHANGE_NONE = "none"' in base_source
    assert (
        'ROUND_STATE_EXCHANGE_CLIENT_METRIC_SUMMARY = "client_metric_summary"'
        in base_source
    )
    assert 'round_state_exchange_name != "none"' not in execution_plan_source
    assert 'NO_ROUND_STATE_EXCHANGE_NAME = "none"' not in executor_source
    assert (
        'CLIENT_METRIC_SUMMARY_EXCHANGE_NAME = "client_metric_summary"'
        not in executor_source
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


def test_agent_runtime_compatibility_does_not_hardcode_privacy_guard_default() -> None:
    path = (
        AGENT_SRC / "services" / "training" / "execution" / "runtime_compatibility.py"
    )

    assert not path.exists(), (
        "agent stored-event runtime compatibility module은 제거됐다. privacy guard "
        "기본값을 agent runtime에 다시 하드코딩하지 않는다.\n"
        f"path={_relative_repo_path(path)}"
    )


def test_round_manager_does_not_own_default_payload_adapter() -> None:
    path = (
        MAIN_SERVER_SRC
        / "services"
        / "federation"
        / "rounds"
        / "round_manager_service.py"
    )
    source = path.read_text(encoding="utf-8")
    imports = _collect_absolute_imports(path)

    assert (
        "main_server.src.services.federation.rounds.payload_adapters.registry"
        not in imports
    )
    assert "build_shared_adapter_round_payload_adapter" not in source
    assert "diagonal_scale" not in source, (
        "RoundManagerService는 round lifecycle orchestration만 소유한다. no-config "
        "legacy payload adapter fallback은 runtime/config profile에 격리하고, "
        "service는 caller가 조립한 payload_adapter를 받는다."
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
    assert 'payload_adapter_name="diagonal_scale"' not in source
    assert "RUNTIME_FALLBACK_SERVER_ROUND_PROFILE" in source, (
        "server runtime config는 live/API fallback 값을 직접 소유하지 않고 "
        "methods.federated_ssl.runtime_fallbacks의 named profile을 읽는다."
    )
    forbidden_active_default_literals = (
        'profile_name="default_peft_classifier.v1"',
        'payload_adapter_kind="peft_classifier"',
        'update_family_name="peft_text_encoder"',
        'aggregation_backend_name="fedavg"',
    )
    violations = [
        snippet for snippet in forbidden_active_default_literals if snippet in source
    ]
    assert not violations, (
        "main_server runtime config는 기본 payload/update/aggregation 선택 문자열을 "
        "직접 하드코딩하지 않는다. live/API compatibility fallback은 "
        "runtime_fallbacks.py의 named profile이 소유한다.\n"
        f"violations={violations}"
    )


def test_round_lifecycle_uses_fallback_profile_for_runtime_defaults() -> None:
    path = (
        MAIN_SERVER_SRC
        / "services"
        / "federation"
        / "rounds"
        / "round_lifecycle_service.py"
    )
    source = path.read_text(encoding="utf-8")

    assert "RUNTIME_FALLBACK_SERVER_ROUND_PROFILE" in source, (
        "RoundLifecycleService는 live/API fallback 값을 직접 소유하지 않는다. "
        "runtime surface 기본값은 methods.federated_ssl.runtime_fallbacks의 "
        "named profile에서 읽는다."
    )
    forbidden_default_literals = (
        'return "fedavg"',
        'or "fedavg"',
        'return "peft_text_encoder"',
        'or "peft_text_encoder"',
    )
    violations = [
        snippet for snippet in forbidden_default_literals if snippet in source
    ]
    assert not violations, (
        "RoundLifecycleService는 update family나 aggregation backend 기본값을 "
        "문자열로 직접 하드코딩하지 않는다.\n"
        f"violations={violations}"
    )


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


def test_live_runtime_layers_do_not_import_concrete_fssl_method_packages() -> None:
    violations: list[tuple[Path, str]] = []
    for root in (AGENT_SRC, MAIN_SERVER_SRC):
        violations.extend(
            _find_forbidden_imports(
                root=root,
                forbidden_prefixes=("methods.federated_ssl.fedmatch",),
            )
        )

    assert not violations, (
        "live agent/main_server는 concrete FL SSL method package를 직접 import하지 "
        "않는다. method identity는 registry/descriptor로 resolve하고, runtime "
        "계층은 capability snapshot과 generic adapter만 해석한다.\n"
        f"{chr(10).join(_live_runtime_import_violation(item) for item in violations)}"
    )


def test_agent_current_task_runner_delegates_runtime_resolution() -> None:
    path = AGENT_SRC / "services" / "training_runtime" / "current_task" / "runner.py"
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "validate_federated_ssl_capability_compatibility",
        "resolve_federated_ssl_method_descriptor",
        "FederatedSslCapabilityPlan",
        "QuerySslObjectiveRuntimeConfig",
        "PEFT_TEXT_ENCODER_UPDATE_FAMILY_NAME",
        "PEFT_CLASSIFIER_UPDATE_PROFILE_NAME",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert "resolve_current_task_runtime" in source
    assert not violations, (
        "agent current-task runner는 orchestration만 소유한다. runtime/profile/"
        "capability 해석은 dispatch.py로 위임해 live task 경계가 runner에 "
        "다시 누적되지 않게 한다.\n"
        f"violations={violations}"
    )


def test_agent_query_ssl_service_delegates_live_fssl_context_parsing() -> None:
    path = (
        AGENT_SRC
        / "services"
        / "training_runtime"
        / "query_ssl"
        / "method_request_builder.py"
    )
    source = path.read_text(encoding="utf-8")
    forbidden_snippets = (
        "def _method_config_from_task_context",
        "def _peer_context_from_task",
        "def _find_peer_context_client_payload",
        "client_contexts",
        "helper_client_ids",
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert "build_method_config_from_live_fssl_context" in source
    assert "build_peer_context_from_live_fssl_context" in source
    assert not violations, (
        "agent query SSL service는 학습 실행만 조립한다. live FSSL context payload "
        "해석은 methods/federated_ssl/live_task_context.py가 소유한다.\n"
        f"violations={violations}"
    )


def test_agent_api_does_not_import_methods_directly() -> None:
    violations = _find_forbidden_imports(
        root=AGENT_SRC / "api",
        forbidden_prefixes=("methods.",),
    )

    assert not violations, (
        "agent API layer는 HTTP request/response 변환만 소유한다. methods-owned "
        "algorithm/runtime surface는 services/training_runtime 또는 inference "
        "adapter 경계에서만 연결한다.\n"
        f"{_format_violations(violations)}"
    )


def test_apps_do_not_import_agent_service_implementations() -> None:
    violations = _find_forbidden_text_snippets(
        root=APPS_SRC,
        snippets=("agent.src.services", "agent/src/services"),
    )

    assert not violations, (
        "apps는 API/contract consumer다. agent service implementation 경로를 "
        "직접 참조하지 않고 generated contract/API client를 통해 통신한다.\n"
        f"{_format_violations(violations)}"
    )


def test_shared_layer_does_not_import_agent_local_contracts() -> None:
    violations = _find_forbidden_imports(
        root=SHARED_SRC,
        forbidden_prefixes=("agent.src",),
    )

    assert not violations, (
        "shared는 공통 contract/domain source of truth다. agent-local contract나 "
        "runtime에 의존하면 shared payload 의미가 local product 표면에 묶인다.\n"
        f"{_format_violations(violations)}"
    )


def test_methods_do_not_import_agent_local_repositories() -> None:
    violations = _find_forbidden_imports(
        root=METHODS_SRC,
        forbidden_prefixes=("agent.src.infrastructure.repositories",),
    )

    assert not violations, (
        "methods는 교체 가능한 algorithm core를 소유한다. raw text/private state "
        "저장소 접근은 agent runtime adapter 경계에 남긴다.\n"
        f"{_format_violations(violations)}"
    )


def test_method_owned_training_core_imports_stay_inside_training_runtime() -> None:
    violations = _find_forbidden_imports(
        root=AGENT_SRC,
        forbidden_prefixes=(
            "methods.adaptation.peft_text_encoder.federated_ssl",
            "methods.adaptation.peft_text_encoder.training",
            "methods.federated_ssl",
            "methods.ssl.runtime",
        ),
        ignored_roots=(AGENT_SRC / "services" / "training_runtime",),
    )

    assert not violations, (
        "agent에서 method-owned local training core를 직접 연결하는 곳은 "
        "services/training_runtime이어야 한다. 다른 feature/runtime 모듈이 "
        "method 의미를 흡수하면 새 method 추가 시 import drift가 생긴다.\n"
        f"{_format_violations(violations)}"
    )


def _live_runtime_import_violation(item: tuple[Path, str]) -> str:
    path, name = item
    return f"- {_relative_repo_path(path)}: {name}"


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
        "FL scripts/runtime adapters는 FedMatch 같은 논문 method "
        "구현을 파일명으로 소유하지 않는다. method identity와 policy 의미는 "
        "methods/federated_ssl/<method>/에 두고, scripts는 entrypoint/report/runtime "
        "bridge만 맡긴다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_fl_scripts_legacy_payload_names_stay_in_compatibility_files() -> None:
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
        Path(
            "scripts/runtime_adapters/federated_agent/generic_client_runtime_bridge.py"
        ),
    }
    actual_paths: set[Path] = set()
    for root in roots:
        for path in _iter_python_files(root):
            source = path.read_text(encoding="utf-8")
            if any(snippet in source for snippet in legacy_snippets):
                actual_paths.add(_relative_repo_path(path))

    assert actual_paths <= allowed_paths, (
        "FL scripts/runtime adapters에 payload-adapter/method legacy 이름을 새 파일로 "
        "확산하지 않는다. 남은 lora_classifier/peft_classifier/FedMatch report "
        "문자열은 docs/contracts/legacy_contract_ledger.md에 기록한 compatibility "
        "표면으로만 허용한다.\n"
        f"{chr(10).join(f'- {path}' for path in sorted(actual_paths - allowed_paths))}"
    )


def test_fl_run_layout_stays_update_family_oriented() -> None:
    path = SCRIPTS_SRC / "experiments" / "fl_ssl" / "support" / "layout.py"
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
        METHODS_SRC / "adaptation" / "peft_text_encoder" / "projection_artifacts.py"
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


def test_peft_text_encoder_does_not_keep_legacy_lora_pass_through_aliases() -> None:
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
        "PeftTextEncoderWithLinearHead =",
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
        "run_lora =",
        "_resolve_lora_backend",
    )
    checked_roots = (
        AGENT_SRC / "services" / "training" / "execution",
        METHODS_SRC / "adaptation" / "peft_text_encoder",
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
        "PEFT text encoder 내부 public surface는 canonical PEFT 이름을 사용한다. "
        "v1 lora_classifier 이름은 shared payload/schema, config compatibility, "
        "artifact field 의미처럼 실제 호환 경계에만 남긴다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_peft_text_encoder_active_module_docs_use_peft_names() -> None:
    checked_paths = (
        PEFT_TEXT_ENCODER_SRC / "initial_state.py",
        PEFT_TEXT_ENCODER_SRC / "update" / "base_state_snapshot.py",
        PEFT_TEXT_ENCODER_SRC / "update" / "json_delta_artifact.py",
        PEFT_TEXT_ENCODER_SRC / "update" / "local_update.py",
        PEFT_TEXT_ENCODER_SRC / "update" / "materialization.py",
        PEFT_TEXT_ENCODER_SRC / "update" / "merged_tensor_artifact.py",
        PEFT_TEXT_ENCODER_SRC / "update" / "partitioned_payload_builder.py",
        PEFT_TEXT_ENCODER_SRC / "update" / "partitioned_tensor_artifact.py",
        PEFT_TEXT_ENCODER_SRC / "training" / "batching.py",
        PEFT_TEXT_ENCODER_SRC / "training" / "optimizer_step.py",
        PEFT_TEXT_ENCODER_SRC / "training" / "partitioned_deltas.py",
        PEFT_TEXT_ENCODER_SRC / "training" / "pseudo_label_diagnostics.py",
        PEFT_TEXT_ENCODER_SRC / "training" / "scalar_metrics.py",
        PEFT_TEXT_ENCODER_SRC / "training" / "step_budget.py",
        PEFT_TEXT_ENCODER_SRC / "federated_ssl" / "partitioned" / "budget.py",
        PEFT_TEXT_ENCODER_SRC / "federated_ssl" / "partitioned" / "training_loop.py",
    )
    forbidden_snippets = (
        "LoRA-classifier",
        "lora_classifier shared payload",
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for path in checked_paths
        for snippet in forbidden_snippets
        if path.exists() and snippet in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "PEFT text encoder active module docstrings/error text는 PEFT 이름을 "
        "사용한다. v1 lora_classifier 이름은 schema constant, legacy factory, "
        "compatibility validator처럼 실제 v1 경계에만 남긴다.\n"
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
        "test-only FL SSL method fixture는 특정 payload-adapter payload contract를 "
        "import하지 않는다. fixture는 method extension seam만 검증해야 한다."
    )
    assert not snippet_violations, (
        "test-only FL SSL method fixture는 concrete payload-adapter 이름을 "
        "하드코딩하지 않는다.\n"
        f"violations={snippet_violations}"
    )


def test_fl_method_descriptor_configs_point_to_real_method_modules() -> None:
    """method descriptor YAML만 먼저 생기는 placeholder config를 막는다."""

    from methods.federated_ssl.method_module_resolution import (
        resolve_federated_ssl_method_family_name,
    )
    from methods.federated_ssl.registry import (
        resolve_federated_ssl_method_descriptor,
        resolve_federated_ssl_method_descriptor_module,
    )

    violations: list[str] = []
    method_package_root = METHODS_SRC / "federated_ssl"
    for config_path in sorted(CONF_FL_METHOD_DESCRIPTOR_SRC.glob("*.yaml")):
        method_name = config_path.stem
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        declared_name = payload.get("name")

        if declared_name != method_name:
            violations.append(
                f"{_relative_repo_path(config_path)}: name={declared_name!r} "
                f"must match filename stem {method_name!r}"
            )
        duplicated_method_metadata_keys = sorted(
            set(payload)
            & {
                "display_name",
                "method_role",
                "implementation_status",
                "original_source",
                "trace_mapping",
                "client_step",
                "server_step",
                "round_state_exchange",
                "report_tags",
                "notes",
            }
        )
        if duplicated_method_metadata_keys:
            violations.append(
                f"{_relative_repo_path(config_path)}: method metadata must stay in "
                "methods/federated_ssl/<method>/descriptor.py, not YAML: "
                f"{duplicated_method_metadata_keys}"
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
        try:
            descriptor = resolve_federated_ssl_method_descriptor(method_name)
            descriptor_module = resolve_federated_ssl_method_descriptor_module(
                method_name
            )
            implementation_family_name = resolve_federated_ssl_method_family_name(
                method_name
            )
        except (ModuleNotFoundError, NotImplementedError, ValueError) as exc:
            violations.append(
                f"{_relative_repo_path(config_path)}: method descriptor is not wired: "
                f"{exc}"
            )
            continue

        if descriptor.name != method_name:
            violations.append(
                f"{_relative_repo_path(config_path)}: resolved descriptor name "
                f"{descriptor.name!r} must match config method name {method_name!r}"
            )
        method_dir = method_package_root / implementation_family_name
        if not method_dir.is_dir():
            violations.append(
                f"{_relative_repo_path(config_path)}: missing "
                f"{_relative_repo_path(method_dir)}"
            )
            continue

        required_files = (
            method_dir / "descriptor.py",
            method_dir / "local_objective.py",
            method_dir / "method_surface.py",
        )
        for required_file in required_files:
            if not required_file.is_file():
                violations.append(
                    f"{_relative_repo_path(config_path)}: missing "
                    f"{_relative_repo_path(required_file)}"
                )
        descriptor_path = Path(descriptor_module.__file__ or "")
        if descriptor_path.name != "descriptor.py" or method_dir not in (
            descriptor_path.parents
        ):
            violations.append(
                f"{_relative_repo_path(config_path)}: descriptor module must be "
                f"owned by {_relative_repo_path(method_dir)}; got "
                f"{_relative_repo_path(descriptor_path)}"
            )
        registry_wiring_shim = method_dir / f"{implementation_family_name}.py"
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
        "capabilities/axes.py에 함께 둔다. 이름/상수만 가진 sibling policy 파일은 "
        "reader path를 늘린다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_peft_partitioned_runtime_uses_query_ssl_policy_predicate() -> None:
    path = PEFT_TEXT_ENCODER_SRC / "federated_ssl" / "partitioned_objective_training.py"
    source = path.read_text(encoding="utf-8")

    assert "is_query_ssl_local_objective_policy" in source
    assert "LOCAL_SSL_POLICIES_FROM_QUERY_SSL" not in source, (
        "active PEFT partitioned runtime은 Query SSL local objective 여부를 "
        "canonical predicate로 확인한다. LOCAL_SSL_POLICIES_FROM_QUERY_SSL 이름은 "
        "compatibility alias로만 남긴다."
    )


def test_federated_ssl_capability_axes_stays_payload_adapter_agnostic() -> None:
    path = METHODS_FEDERATED_SSL_SRC / "capabilities" / "axes.py"
    imports = _collect_absolute_imports(path)
    forbidden_imports = {
        "shared.src.contracts.adapter_contract_families.classifier_head",
        "shared.src.contracts.adapter_contract_families.diagonal_scale",
        "shared.src.contracts.adapter_contract_families.lora_classifier",
    }
    source = path.read_text(encoding="utf-8")

    assert not sorted(imports & forbidden_imports), (
        "FL SSL capability axis는 local/server policy 이름만 소유한다. "
        "payload-adapter payload contract나 runtime backend 해석은 "
        "methods/adaptation/<family>/federated_ssl/가 소유한다."
    )
    assert "lora_classifier" not in source, (
        "capabilities/axes.py는 LoRA-classifier family literal을 하드코딩하지 않는다."
    )


def test_federated_ssl_hooks_stay_method_agnostic() -> None:
    hook_root = METHODS_FEDERATED_SSL_SRC / "hooks"
    forbidden_snippets = ("fedmatch", "sigma", "psi")
    violations: list[str] = []
    for path in _iter_python_files(hook_root):
        source = path.read_text(encoding="utf-8").lower()
        for snippet in forbidden_snippets:
            if snippet in source:
                violations.append(f"{_relative_repo_path(path)}: {snippet}")

    assert not violations, (
        "methods/federated_ssl/hooks는 여러 FL SSL method가 공유할 hook surface만 "
        "소유한다. FedMatch method 이름과 sigma/psi 같은 method-local partition "
        "의미는 methods/federated_ssl/<method>/ 아래에 둔다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_federated_ssl_method_packages_do_not_own_payload_adapter_runtime_files() -> (
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
        "LoRA-classifier/full encoder/DoRA 같은 payload-adapter 실행 구현은 "
        "methods/adaptation/<family>/federated_ssl/에 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_payload_adapter_federated_ssl_files_do_not_multiply_by_method_name() -> None:
    method_fragments = (
        "fedmatch",
        "fixmatch",
        "flexmatch",
        "freematch",
        "dash",
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
        "methods/adaptation/<family>/federated_ssl/는 payload-adapter 실행 primitive를 "
        "소유한다. 새 FL SSL method마다 <method>_*.py 파일을 늘리지 말고 "
        "method 의미는 methods/federated_ssl/<method>/에 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_peft_text_encoder_partitioned_runtime_is_method_neutral() -> None:
    checked_paths = (
        METHODS_SRC
        / "adaptation"
        / "peft_text_encoder"
        / "federated_ssl"
        / "partitioned"
        / "training_loop.py",
        METHODS_SRC
        / "adaptation"
        / "peft_text_encoder"
        / "federated_ssl"
        / "partitioned_objective_training.py",
    )
    import_violations: list[str] = []
    snippet_violations: list[str] = []
    for path in checked_paths:
        import_violations.extend(
            f"{_relative_repo_path(path)}: {imported}"
            for imported in _collect_absolute_imports(path)
            if imported.startswith("methods.federated_ssl.fedmatch")
        )
        source = path.read_text(encoding="utf-8").lower()
        snippet_violations.extend(
            f"{_relative_repo_path(path)}: {snippet}"
            for snippet in ("fedmatch", "sigma", "psi")
            if snippet in source
        )

    assert not import_violations and not snippet_violations, (
        "partitioned PEFT text encoder runtime은 payload-adapter execution "
        "primitive다. FedMatch objective, metric prefix, partition 이름은 "
        "methods/federated_ssl/fedmatch/의 caller가 주입해야 한다.\n"
        f"{chr(10).join(f'- import: {item}' for item in import_violations)}"
        f"{chr(10).join(f'- snippet: {item}' for item in snippet_violations)}"
    )


def test_methods_lora_classifier_compatibility_package_is_removed() -> None:
    legacy_root = METHODS_SRC / "adaptation" / "lora_classifier"
    existing_paths = _existing_non_cache_paths((legacy_root,))

    assert not existing_paths, (
        "methods/adaptation/lora_classifier는 더 이상 internal compatibility "
        "package로 유지하지 않는다. 구현 source of truth는 "
        "methods/adaptation/peft_text_encoder/**이고, lora_classifier 이름은 "
        "old artifact/report reader compatibility 표면에만 남긴다.\n"
        f"{chr(10).join(f'- {path}' for path in existing_paths)}"
    )


def test_shared_lora_classifier_v1_contract_is_removed() -> None:
    legacy_contract = (
        SHARED_SRC / "contracts" / "adapter_contract_families" / "lora_classifier.py"
    )
    legacy_fixture = (
        REPO_ROOT / "tests" / "contracts" / "fixtures" / "lora_classifier_delta.v1.json"
    )

    assert not legacy_contract.exists(), (
        "v1 lora_classifier shared parser/factory는 active shared contract 표면에서 "
        "제거된 상태를 유지한다. 과거 artifact는 report/materialization old-reader "
        "경계에서만 canonical PEFT 표면으로 정규화한다."
    )
    assert not legacy_fixture.exists(), (
        "golden fixture는 active shared payload shape만 보존한다. v1 lora payload "
        "fixture를 재도입하면 shared contract가 다시 legacy producer를 소유하게 된다."
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
        "package다. 새 internal code는 peft_text_encoder 경로를 직접 import한다.\n"
        f"{_format_violations(violations)}"
    )


def test_peft_text_encoder_does_not_import_fedmatch_method() -> None:
    violations = _find_forbidden_imports(
        root=PEFT_TEXT_ENCODER_SRC,
        forbidden_prefixes=("methods.federated_ssl.fedmatch",),
    )

    assert not violations, (
        "methods/adaptation/peft_text_encoder/**는 PEFT text encoder 실행 "
        "primitive를 "
        "소유한다. FedMatch 의미, partition routing, original parameter는 "
        "methods/federated_ssl/fedmatch/에서 callable/config로 주입한다.\n"
        f"{_format_violations(violations)}"
    )


def test_peft_text_encoder_does_not_depend_on_legacy_lora_classifier() -> None:
    violations = _find_forbidden_imports(
        root=PEFT_TEXT_ENCODER_SRC,
        forbidden_prefixes=(
            "methods.adaptation.classifier_head",
            "methods.adaptation.lora_classifier",
        ),
    )

    assert not violations, (
        "새 peft_text_encoder 내부 코드는 legacy classifier_head/lora_classifier "
        "경로를 import하지 않는다. 기존 경로는 migration shim으로만 남기고, 내부 "
        "source of truth는 peft_text_encoder 아래에 둔다.\n"
        f"{_format_violations(violations)}"
    )


def test_classification_adaptation_is_modality_independent() -> None:
    violations = _find_forbidden_imports(
        root=LINEAR_HEAD_CLASSIFICATION_SRC,
        forbidden_prefixes=(
            "methods.adaptation.classifier_head",
            "methods.adaptation.lora_classifier",
            "methods.adaptation.peft_text_encoder",
            "methods.adaptation.text_classifier",
        ),
    )

    assert not violations, (
        "methods/classification/linear_head/**는 modality-independent classification "
        "primitive를 소유한다. text-specific PEFT encoder나 legacy classifier_head "
        "경로를 import하지 않는다.\n"
        f"{_format_violations(violations)}"
    )


def test_peft_text_encoder_uses_peft_adapters_axis() -> None:
    violations = _find_forbidden_imports(
        root=PEFT_TEXT_ENCODER_SRC,
        forbidden_prefixes=(
            "methods.adaptation.lora.",
            "methods.adaptation.peft.",
        ),
    )

    assert not violations, (
        "PEFT encoder text encoder/head는 LoRA/DoRA mechanism을 "
        "methods/adaptation/peft_adapters/** 축으로만 참조한다. legacy "
        "methods/adaptation/lora 또는 methods/adaptation/peft 경로에 묶지 않는다.\n"
        f"{_format_violations(violations)}"
    )


def test_peft_adapter_mechanisms_are_not_trainable_state_family_leaves() -> None:
    update_family_dir = (
        CONF_SRC / "strategy_axes" / "model_architecture" / "update_family"
    )
    mechanism_fragments = ("lora", "rslora", "dora")
    violations = [
        _relative_repo_path(config_path)
        for config_path in sorted(update_family_dir.glob("*.yaml"))
        if any(fragment in config_path.stem.lower() for fragment in mechanism_fragments)
    ]

    assert not violations, (
        "LoRA/RSLoRA/DoRA는 PEFT adapter mechanism이지 trainable state/update "
        "family가 아니다. mechanism 선택은 strategy_axes/model_architecture/peft와 "
        "methods/adaptation/peft_adapters/<mechanism>/builder.py에 두고, "
        "trainable_state/update_family에는 peft_text_encoder/prototype_pack 같은 "
        "공유 상태 family만 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
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
        "legacy shared compatibility 값이 남아 있더라도 methods-level 구현 폴더나 "
        "trainable_state/update_family leaf를 다시 만들지 않는다.\n"
        f"{chr(10).join(f'- {path}' for path in existing_paths)}"
    )


def test_active_docs_do_not_show_lora_classifier_as_current_fl_verifier() -> None:
    checked_paths = (
        CONF_SRC / "README.md",
        REPO_ROOT / "apps" / "experiment_dashboard" / "README.md",
        MAIN_SERVER_SRC / "services" / "federation" / "rounds" / "README.md",
        SCRIPTS_SRC / "README.md",
        SCRIPTS_SRC / "experiments" / "fl_ssl" / "federated_simulation" / "README.md",
        CONF_SRC / "strategy_axes" / "fl" / "README.md",
        REPO_ROOT / "docs" / "project_execution_plan.md",
        REPO_ROOT / "docs" / "operations" / "local-runbook.md",
    )
    forbidden_snippets = (
        "legacy fallback",
        "PEFT-classifier",
        "PEFT text-classifier",
        "--expected-payload-adapter-kind lora_classifier",
        "--expect-peft-classifier-aggregate-snapshot",
        "expect_peft_classifier_aggregate_snapshot",
        "--expect-peft-encoder-aggregate-snapshot",
        "expect_peft_encoder_aggregate_snapshot",
        "--expect-lora-classifier-aggregate-snapshot",
        "FedAvg + FixMatch + LoRA-classifier",
        "LoRA-classifier simulation 병목",
        "method-owned LoRA-classifier",
        "LoRA-classifier",
        "lora_classifier model builder",
        "LoRA-classifier `partitioned_delta_average`",
        "lora_classifier leaf",
        "round_runtime.payload_adapter_name",
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for path in checked_paths
        for snippet in forbidden_snippets
        if path.exists() and snippet in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "active config/runbook 문서는 현재 PEFT text encoder 실행 용어를 "
        "사용한다. lora_classifier verifier flag, PEFT-classifier, "
        "LoRA-classifier 표기는 legacy audit/contract 문서에만 남긴다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_active_docs_use_current_trainable_state_vocabulary() -> None:
    checked_paths = (
        REPO_ROOT / "agent" / "README.md",
        REPO_ROOT / "agent" / "src" / "services" / "README.md",
        REPO_ROOT / "docs" / "ai_context_manifest.yaml",
        REPO_ROOT / "docs" / "contracts" / "model_manifest_v1.md",
        REPO_ROOT
        / "docs"
        / "contracts"
        / "central_peft_text_encoder_trainer_contract.md",
        REPO_ROOT / "docs" / "contracts" / "shared_adapter_contracts_v1.md",
        REPO_ROOT / "docs" / "contracts" / "training_task_v1.md",
        REPO_ROOT / "docs" / "contracts" / "training_update_envelope_v1.md",
    )
    forbidden_snippets = (
        "global classifier",
        "central PEFT classifier",
        "PEFT classifier trainer",
        "PEFT classifier scaffold",
        "PEFT + text classifier",
    )
    violations = [
        f"{_relative_repo_path(path)}: {snippet}"
        for path in checked_paths
        for snippet in forbidden_snippets
        if path.exists() and snippet in path.read_text(encoding="utf-8")
    ]

    assert not violations, (
        "active agent/contract 문서는 classifier 전용 구조처럼 읽히지 않도록 "
        "shared scoring state, PEFT text encoder, trainable state family 용어를 "
        "사용한다. historical ledger나 legacy compatibility 문서만 과거 이름을 "
        "해석할 수 있다.\n"
        f"{chr(10).join(f'- {violation}' for violation in violations)}"
    )


def test_active_surface_and_runbook_docs_stay_concise() -> None:
    checked_paths = (
        CONF_SRC / "README.md",
        SCRIPTS_SRC / "README.md",
        REPO_ROOT / "docs" / "project_execution_plan.md",
        SCRIPTS_SRC / "experiments" / "central" / "ssl_control" / "README.md",
        SCRIPTS_SRC / "experiments" / "fl_ssl" / "README.md",
        SCRIPTS_SRC / "experiments" / "fl_ssl" / "federated_simulation" / "README.md",
    )
    max_lines_by_path = {
        CONF_SRC / "README.md": 160,
        SCRIPTS_SRC / "README.md": 120,
        REPO_ROOT / "docs" / "project_execution_plan.md": 160,
        SCRIPTS_SRC / "experiments" / "central" / "ssl_control" / "README.md": 100,
        SCRIPTS_SRC / "experiments" / "fl_ssl" / "README.md": 160,
        SCRIPTS_SRC
        / "experiments"
        / "fl_ssl"
        / "federated_simulation"
        / "README.md": 120,
    }
    violations = [
        (
            _relative_repo_path(path),
            len(path.read_text(encoding="utf-8").splitlines()),
            max_lines_by_path[path],
        )
        for path in checked_paths
        if len(path.read_text(encoding="utf-8").splitlines()) > max_lines_by_path[path]
    ]

    assert not violations, (
        "surface map과 script runbook은 active 진입점이다. 긴 완료 이력, 특정 run "
        "세부, 반복 cookbook은 docs/notes archive로 내리고 active 문서는 현재 "
        "경계와 read path만 유지한다.\n"
        f"{chr(10).join(_format_doc_line_violation(item) for item in violations)}"
    )


def _format_doc_line_violation(violation: tuple[Path, int, int]) -> str:
    path, count, limit = violation
    return f"- {path}: {count}>{limit}"


def test_fl_round_e2e_does_not_exercise_removed_diagonal_scale_runtime() -> None:
    path = REPO_ROOT / "tests" / "integration" / "test_fl_round_e2e.py"
    imports = _collect_absolute_imports(path)
    source = path.read_text(encoding="utf-8")

    assert (
        "shared.src.contracts.adapter_contract_families.diagonal_scale" not in imports
    )
    assert "diagonal_scale" not in source, (
        "root FL round E2E는 현재 update family runtime을 검증한다. diagonal_scale는 "
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
        PEFT_TEXT_ENCODER_AGGREGATION_SRC,
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
        "classification/peft_text_encoder aggregation 계층은 family state를 generic "
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
            "methods.adaptation.peft_text_encoder",
            "methods.adaptation.text_classifier",
            "shared.src.contracts.adapter_contract_families.classifier_head",
            "shared.src.contracts.adapter_contract_families.lora_classifier",
        ),
    )

    assert not violations, (
        "methods/adaptation/peft_adapters/**는 LoRA/DoRA 같은 PEFT mechanism만 "
        "소유한다. classifier label, task head, update payload 의미는 "
        "peft_text_encoder adaptation 또는 shared contract가 소유한다.\n"
        f"{_format_violations(violations)}"
    )


def test_peft_adapter_registry_does_not_hardcode_concrete_mechanisms() -> None:
    source = (PEFT_ADAPTERS_SRC / "registry.py").read_text(encoding="utf-8")
    forbidden_snippets = (
        "from methods.adaptation.peft_adapters.lora import",
        "builder as _lora_adapter",
        'hasattr(cfg, "lora")',
        "return cfg.lora",
        'return "lora"',
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "PEFT adapter registry는 registry primitive와 convention import만 소유한다. "
        "LoRA/DoRA 같은 concrete mechanism import 목록이나 legacy cfg.lora "
        "fallback은 active registry에 누적하지 않는다.\n"
        f"violations={violations}"
    )


def test_simulation_inline_peft_delta_does_not_default_adapter_mechanism() -> None:
    source = (
        PEFT_TEXT_ENCODER_SRC / "update" / "simulation_inline_delta.py"
    ).read_text(encoding="utf-8")
    forbidden_snippets = (
        'getattr(config, "peft_adapter_name", "lora")',
        'or "lora"',
    )
    violations = [snippet for snippet in forbidden_snippets if snippet in source]

    assert not violations, (
        "simulation inline delta는 PEFT adapter mechanism 기본값을 재소유하지 않는다. "
        "실행 config의 to_peft_adapter_config_payload()에서 canonical "
        "peft_adapter_name을 읽어야 한다.\n"
        f"violations={violations}"
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


def test_agent_legacy_training_package_is_not_reintroduced() -> None:
    package_root = AGENT_SRC / "services" / "training"

    assert not package_root.exists(), (
        "agent/src/services/training은 stored-event pseudo-label self-training "
        "legacy package다. 현재 runtime은 services/training_runtime에서 current "
        "TrainingTask와 Query SSL/FSSL local objective adapter만 소유한다.\n"
        f"path={_relative_repo_path(package_root)}"
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
        "privacy guard 정책과 payload-adapter별 clipping 계산은 "
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
            "methods.adaptation.peft_text_encoder",
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


def test_agent_scoring_backends_do_not_keep_payload_family_modules() -> None:
    package_root = AGENT_SRC / "services" / "inference" / "scoring_backends"
    forbidden_fragments = ("classifier_head", "diagonal_scale", "lora_classifier")
    violations = [
        _relative_repo_path(path)
        for path in _iter_python_files(package_root)
        if any(fragment in path.stem for fragment in forbidden_fragments)
    ]

    assert not violations, (
        "payload/update-family별 scoring core는 "
        "methods/adaptation/<family>가 소유한다. "
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
        "server update materialization dispatcher는 payload family별 contract를 "
        "직접 알지 않는다. family-specific preflight는 "
        "methods/adaptation/<family>/server_preflight.py에 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )
    assert "agent-local://" not in source, (
        "agent-local artifact ref 정책은 dispatcher가 아니라 해당 payload family가 "
        "소유한다."
    )
    assert "peft_classifier" not in source, (
        "dispatcher는 PEFT text encoder payload kind도 하드코딩하지 않는다. "
        "패키지 경로 alias는 구현 owner 옆 payload_adapter_module manifest가 소유한다."
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
        "server update compatibility dispatcher는 payload family별 "
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
        "runtime/objective compatibility dispatcher는 payload family별 "
        "contract를 직접 알지 않는다. family-specific 검증은 "
        "methods/adaptation/<family>/runtime_compatibility.py에 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )
    assert "lora_classifier" not in source, (
        "dispatcher는 LoRA-classifier family 이름을 하드코딩하지 않는다."
    )
    assert "peft_classifier" not in source, (
        "dispatcher는 PEFT text encoder payload kind도 하드코딩하지 않는다. "
        "패키지 경로 alias는 구현 owner 옆 payload_adapter_module manifest가 소유한다."
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
        "FL simulation runtime compatibility adapter는 payload adapter literal로 "
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
        "FL SSL server update dispatcher는 payload adapter별 payload contract를 "
        "직접 알지 않는다. family-specific backend 해석은 "
        "methods/adaptation/<family>/federated_ssl/server_update_policy.py가 "
        "소유한다."
    )
    assert "lora_classifier" not in source, (
        "FL SSL server update dispatcher는 LoRA-classifier family 이름을 "
        "하드코딩하지 않는다."
    )
    assert "peft_classifier" not in source, (
        "FL SSL server update dispatcher는 PEFT text encoder payload kind도 "
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
        "methods/adaptation/peft_text_encoder/** owner 경계에 둔다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_query_peft_run_artifacts_do_not_keep_writer_exporter_monolith() -> None:
    orchestrator_path = QUERY_SSL_TEXT_ENCODER_IO_SRC / "artifacts.py"
    expected_responsibility_files = (
        QUERY_SSL_TEXT_ENCODER_IO_SRC / "artifact_paths.py",
        QUERY_SSL_TEXT_ENCODER_IO_SRC / "artifact_writer.py",
        QUERY_SSL_TEXT_ENCODER_IO_SRC / "manifest_builder.py",
        QUERY_SSL_TEXT_ENCODER_IO_SRC / "model_artifact_exporter.py",
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


def test_query_peft_teacher_pseudo_label_export_surface_is_removed() -> None:
    legacy_exporter_path = (
        QUERY_SSL_TEXT_ENCODER_IO_SRC / "teacher_pseudo_label_exporter.py"
    )
    legacy_builder_path = (
        QUERY_SSL_TEXT_ENCODER_IO_SRC / "teacher_pseudo_label_builder.py"
    )
    legacy_algorithm_path = (
        QUERY_SSL_TEXT_ENCODER_CONFIG_SRC / "pseudo_label_algorithm.py"
    )
    methods_builder_path = METHODS_SSL_SRC / "teacher_pseudo_label.py"
    writer_path = (
        QUERY_SSL_TEXT_ENCODER_IO_SRC / "teacher_pseudo_label_artifact_writer.py"
    )
    removed_paths = (
        legacy_exporter_path,
        legacy_builder_path,
        legacy_algorithm_path,
        methods_builder_path,
        writer_path,
    )
    existing_paths = [path for path in removed_paths if path.exists()]

    assert not existing_paths, (
        "teacher pseudo-label export는 active 중앙 online SSL workflow가 아니다.\n"
        f"{chr(10).join(f'- {_relative_repo_path(path)}' for path in existing_paths)}"
    )


def test_scripts_reporting_does_not_wrap_shared_classification_report() -> None:
    facade_path = SCRIPTS_SRC / "reporting" / "classification_report.py"

    assert not facade_path.exists(), (
        "classification report canonical utility는 shared domain service가 소유한다. "
        "scripts/support/reporting에는 단순 re-export wrapper를 두지 않는다.\n"
        f"facade path={_relative_repo_path(facade_path)}"
    )


def test_scripts_do_not_wrap_shared_labeled_query_rows() -> None:
    facade_path = SCRIPTS_SRC / "io" / "labeled_query_rows.py"

    assert not facade_path.exists(), (
        "labeled query row canonical contract는 shared contract가 소유한다. "
        "scripts/io에는 단순 re-export wrapper를 두지 않는다.\n"
        f"facade path={_relative_repo_path(facade_path)}"
    )


def test_main_server_round_payload_adapter_package_has_no_concrete_modules() -> None:
    package_root = (
        MAIN_SERVER_SRC / "services" / "federation" / "rounds" / "payload_adapters"
    )
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
        "main_server round payload adapter package는 shared adapter payload registry와 "
        "aggregation backend를 generic runtime으로 조합한다. concrete payload adapter "
        "module은 추가하지 않는다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_scripts_do_not_import_removed_main_server_round_family_package() -> None:
    legacy_import = "main_server.src.services.federation.rounds.families"
    violations: list[tuple[Path, str]] = []
    for root in (SCRIPTS_SRC, REPO_ROOT / "tests"):
        violations.extend(
            _find_forbidden_imports(
                root=root,
                forbidden_prefixes=(legacy_import,),
            )
        )

    assert not violations, (
        "scripts/tests는 제거된 main_server rounds.families package를 import하지 "
        "않는다. server round wiring은 payload_adapters package와 generic "
        "payload_adapter field를 사용한다.\n"
        f"{_format_violations(violations)}"
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
        "payload-adapter projection은 methods/federated/aggregation이 소유한다.\n"
        f"{chr(10).join(f'- {path}' for path in violations)}"
    )


def test_main_server_aggregation_package_has_no_method_or_payload_literals() -> None:
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
        "generic boundary만 둔다. aggregation method나 payload adapter 상세 문자열은 "
        "methods/ 쪽 strategy/projection에 둔다.\n"
        f"{chr(10).join(f'- {path}: {snippet}' for path, snippet in violations)}"
    )


def test_main_server_aggregation_methods_do_not_define_payload_specific_services() -> (
    None
):
    package_root = (
        MAIN_SERVER_SRC / "services" / "federation" / "rounds" / "aggregation"
    )

    payload_adapter_prefixes = {
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
                node.name.startswith(prefix) for prefix in payload_adapter_prefixes
            ) and (
                node.name.endswith("AggregationService")
                or node.name.endswith("AggregationConfig")
            ):
                violations.append((_relative_repo_path(path), node.name))

    assert not violations, (
        "main_server aggregation method file은 payload-adapter별 service/config "
        "class를 누적하지 않는다. payload adapter 차이는 shared payload contract와 "
        "generic runtime spec 뒤에 둔다.\n"
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


def test_payload_adapter_fedavg_modules_live_under_aggregation_package() -> None:
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
        "payload adapter별 FedAvg core/projection은 root 수평 파일이 아니라 "
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
        "소유한다. payload adapter별 FedAvg core와 payload projection은 "
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
        "partitioned delta 평균은 payload-adapter payload 해석이 먼저 필요한 "
        "backend다. registry convention만 만족시키는 "
        "methods/federated/aggregation/partitioned_* 얇은 package를 만들지 "
        "않는다.\n"
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
        "사용하지 않는다. server-owned artifact lifecycle은 좁은 capability "
        "package에 둔다.\n"
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
