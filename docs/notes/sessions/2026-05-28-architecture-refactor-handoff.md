# 2026-05-28 architecture refactor handoff

## 최종 목표

프로젝트 전체에서 강결합과 하드코딩을 줄이고, 새 method/backend/family를 추가할 때
수정 위치가 `methods/`와 `conf/` 중심으로 예측되는 구조로 간다. `scripts/`는 실행,
sweep, report, visualization thin wrapper만 맡고, `agent`/`main_server`는 runtime
adapter와 orchestration만 맡는다. 기존 MD 문서는 과거 계획일 수 있으므로
`docs/ai_context_manifest.yaml`, `docs/execution_index.md`, 코드, contract, test를
현재 증거로 다시 확인해야 한다. `data`, `runs`, `main_server/state` 등 실험 자료는
건드리지 않는다.

## 완료 및 push된 작업

최근 push된 기준 커밋은 `2924490a refactor: registry discovery와 exchange 계약 소유권 정리`다.
그 전 주요 커밋은 `2691a978 refactor: PEFT text encoder 실행 표면 정리`와
`a44e6377 refactor: runtime fallback profile 경계 정리`다.

완료된 방향:

- `train_peft_*_classifier` 실행 표면을 `run_peft_*_control`로 정리했다.
- `central_peft_classifier_trainer_contract.md`를
  `central_peft_text_encoder_trainer_contract.md`로 바꿨다.
- active 문서의 `global classifier`, `PEFT classifier scaffold` 같은 현재형 표현을
  `shared scoring state`, `PEFT text encoder`, `trainable state family` 중심으로 정리했다.
- `noop` privacy guard 기본값은 runtime fallback profile에서 읽게 했다.
- client diagnostics와 PEFT adapter builder는 concrete 목록이 아니라 convention
  discovery로 읽게 했다.
- `round_state_exchange`의 `none`, `client_metric_summary` 이름은
  `methods/federated_ssl/base.py`가 소유하게 했다.

## 현재 미커밋 변경

현재 worktree에는 FedMatch/FixMatch 관련 compatibility 정리 변경이 남아 있다.
목적은 generic `methods/federated_ssl/compatibility.py`가 FedMatch-specific
`fedmatch_partitioned`, `fedmatch_agreement`, `fixmatch` 허용 조합을 직접 소유하지
않게 하는 것이다. 해당 규칙은 `methods/federated_ssl/fedmatch/compatibility.py`로
이동했다. `scripts/experiments/fl_ssl/federated_simulation/simulation.py`는
`validate_federated_ssl_simulation_runtime_support(..., method_descriptor=...)`를
넘겨 method-local validator가 실행되도록 바뀌었다.

검증된 명령:

- `uv run ruff check methods/federated_ssl/compatibility.py methods/federated_ssl/fedmatch/compatibility.py scripts/experiments/fl_ssl/federated_simulation/simulation.py tests/unit/test_methods_federated_capabilities.py tests/architecture/test_layer_dependencies.py`
- `uv run pytest tests/unit/test_methods_federated_capabilities.py tests/architecture/test_layer_dependencies.py::test_generic_fl_ssl_compatibility_does_not_own_fedmatch_policy_rules`
- `uv run pytest tests/unit/test_run_federated_simulation.py::test_run_simulation_request_rejects_manual_partitioned_update_until_producer tests/unit/test_run_federated_simulation.py::test_method_owned_peft_round_uses_method_trainer_before_manual_query_ssl tests/unit/test_run_federated_simulation.py::test_run_simulation_request_completes_peft_classifier_inline_delta_round`

## 다음 작업

1. 현재 미커밋 변경을 한 번 더 확인하고, 필요하면 선택 테스트를 재실행한 뒤 커밋/푸시한다.
2. `scripts/runtime_adapters/**`에 남은 family-specific import가 runtime bridge 수준인지,
   아니면 method/family 의미를 재소유하는지 점검한다.
3. report verifier/dashboard/result-index의 PEFT/FedMatch 기대값이 과거 run
   compatibility인지 active 실행 의미인지 분리한다.
4. active docs는 source of truth가 아니라 현재 코드와 contract를 설명하는 보조 계층으로만
   보고, 오래된 계획 문서는 `docs/notes/**`로 격리한다.
