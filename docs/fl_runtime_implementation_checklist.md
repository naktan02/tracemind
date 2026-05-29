# FL Runtime Implementation Checklist

이 문서는 FL runtime translation에서 아직 확인해야 하는 active gate만 둔다.
완료된 세부 구현 이력과 과거 run path는
`docs/notes/decisions/2026-05-28-archived-fl-runtime-implementation-checklist.md`에
보관했다.

구조 판단은 `docs/architecture/target-method-runtime-structure.md`를 우선한다.
이 문서가 코드, contract, code-adjacent README와 충돌하면 코드 가까운 source of
truth를 먼저 확인한다.

## 고정 경계

- `methods/`는 FL SSL method, aggregation/update policy, sampling/diagnostics core를
  소유한다.
- `conf/`는 Hydra 실행 조합과 parameter leaf만 소유한다.
- `scripts/experiments/fl_ssl`는 entrypoint, simulation orchestration, report/artifact
  writer만 소유한다.
- `agent`와 `main_server`는 method 이름이 아니라 runtime capability와 selected core를
  연결한다.
- `shared`는 cross-boundary contract와 canonical payload 해석만 소유한다.

## 완료로 유지할 invariant

- 새 FL SSL method 추가의 기본 변경 위치는
  `methods/federated_ssl/<method>/`, `conf/strategy_axes/fssl_method`,
  필요한 capability config, tests다.
- `scripts`는 `fedmatch_agreement`, `fedmatch_partitioned`, `lora_classifier`,
  `diagonal_scale` 같은 method/update-family legacy 이름으로 실행 분기하지 않는다.
- aggregation weight, diagnostic/probe sampling, labeled row canonical label 해석은
  각각 `methods/federated`, `methods/federated_ssl`, `shared` helper를 통해 읽는다.
- `lora_classifier`와 `diagonal_scale` 구현 package는 제거된 상태를 유지한다.
- 과거 report/artifact ingest가 필요하면 compatibility reader에서만 처리하고,
  새 producer나 active config에 legacy 이름을 되살리지 않는다.

## 남은 gate

- winner method가 요구하는 shared family/state/update payload를 확정한다.
- `gpu_local + mxbai` metadata가 있는 current 30-round main/sweep 산출물을 새 protocol로
  남긴다.
- final stress split은 Dirichlet `alpha=0.1`, `30 rounds`로 별도 명시 실행한다.
- live `agent`/`main_server` runtime translation은 simulation winner가 확정된 뒤
  capability adapter 단위로 진행한다.

## 검증

- architecture guard:
  `tests/architecture/test_fl_simulation_runtime_boundaries.py`,
  `tests/architecture/test_layer_dependencies.py`
- FL simulation unit/integration surface:
  `tests/unit/test_run_federated_simulation.py`,
  `tests/unit/test_methods_federated_capabilities.py`,
  `tests/unit/test_methods_federated_ssl.py`
- report/artifact compatibility:
  `scripts/experiments/fl_ssl/verify_federated_report_artifacts.py`와
  `docs/operations/fl_ssl_artifact_verification_manifest.current.json`
