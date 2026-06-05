# Experiment Orchestration Readability Refactor Plan

## 목적

중앙 supervised control, 중앙 SSL control, FL SSL/FSSL 실행 경로를 사람이 같은
언어로 읽을 수 있게 정리한다. 목표는 새 framework를 만드는 것이 아니라, 각
entrypoint의 첫 읽기 경로가 다음 흐름으로 드러나게 하는 것이다.

```text
Hydra cfg
-> typed request
-> runtime preparation
-> execution loop
-> artifact/report
```

학습 의미, split 의미, method/algorithm 계산, artifact schema, output layout은
바꾸지 않는다.

## 적용 범위

우선 읽기 표면을 맞출 대상:

- `scripts/experiments/central/ssl_control/run_peft_supervised_control.py`
- `scripts/experiments/central/ssl_control/run_peft_ssl_control.py`
- `scripts/experiments/fl_ssl/run_federated_simulation.py`
- `scripts/support/query_ssl_text_encoder/runners/*`
- `scripts/experiments/fl_ssl/federated_simulation/*`
- `scripts/runtime_adapters/*`

계산 core는 원칙적으로 리팩터링 대상이 아니다:

- `methods/ssl/*`
- `methods/federated_ssl/*`
- `methods/adaptation/peft_text_encoder/*`

core를 건드려야 할 때는 orchestration readability 작업이 아니라 별도 기능/성능
변경으로 분리한다.

## 변경하지 않는 것

- SSL/FSSL method 의미
- PEFT/local training loop 의미
- dataset split/materialized data 의미
- aggregation/update/payload 의미
- artifact/report/manifest schema
- output path slug 규칙
- Hydra 값의 의미
- `shared`, `methods`, `conf`, `agent`, `main_server`, `scripts` 소유 경계

## 공통 목표 모양

각 entrypoint는 가능하면 다음 순서로 읽히게 한다.

```python
def main(cfg):
    if run_sweep_if_requested(cfg):
        return

    output_dir = resolve_output_dir(cfg)
    request = build_request_from_config(cfg, output_dir=output_dir)
    result = run_experiment_request(request)
    print_result_summary(output_dir=output_dir, result=result)
```

이 모양을 강제하는 공통 base class나 facade는 만들지 않는다. 중앙 SSL과 FSSL은
같은 vocabulary를 쓰되, 각 실행 레일의 local helper로 정리한다.

## 단계별 진행

### 0단계: 열린 기능 변경 닫기

읽기 리팩터링 전에 현재 성능/안정화 변경을 별도 커밋으로 닫는다.

현재 열린 변경:

- FedMatch/FSSL round boundary transient resource cleanup

성공 기준:

```bash
uv run pytest tests/unit/test_run_federated_simulation.py \
  tests/unit/test_scripts_hydra_configs.py \
  tests/unit/test_simulation_report_builder.py \
  tests/unit/test_federated_agent_runtime_adapters.py
```

### 1단계: 실행 경로 inventory 작성

코드 이동 없이 현재 경로를 표로 정리한다.

각 entrypoint마다 확인할 항목:

```text
entrypoint
typed request 위치
runtime preparation 위치
execution loop 위치
artifact/report writer 위치
method-specific 분기 위치
처음 읽는 데 필요한 주요 파일 수
```

산출물:

```text
docs/architecture/experiment-orchestration-readability-inventory.md
```

성공 기준:

- 중앙 supervised, 중앙 SSL, FSSL이 같은 표에 들어간다.
- FSSL 전용 개선안으로 빠지지 않는다.
- 코드 변경은 하지 않는다.

### 2단계: baseline snapshot tests 추가

리팩터링 전후 실행 계약이 변하지 않는지 고정한다.

후보 테스트:

- 중앙 supervised control compose/request snapshot
- 중앙 SSL control compose/request snapshot
- FSSL main/smoke request snapshot

고정할 값 예시:

```text
dataset/split/seed
local_epochs, batch_size, max_steps
objective/method name
update family
aggregation backend
output layout slug 핵심 값
```

성공 기준:

```bash
uv run pytest tests/unit/test_scripts_hydra_configs.py
```

상태:

- 완료: 중앙 supervised smoke, 중앙 Query SSL smoke, FSSL smoke request의
  orchestration contract snapshot을 `tests/unit/test_scripts_hydra_configs.py`에
  추가했다.

### 3단계: entrypoint 첫 화면 정리

대상:

- 중앙 supervised entrypoint
- 중앙 SSL entrypoint
- FSSL entrypoint

목표:

- sweep/single-run 분기
- output dir 결정
- request 생성
- runner 호출
- 결과 출력

위 흐름이 entrypoint 첫 화면에 보이게 한다.

주의:

- Hydra config 의미를 Python helper에 복제하지 않는다.
- output path와 result 출력 형식을 바꾸지 않는다.
- 각 entrypoint별 helper를 우선하고 공통 framework를 만들지 않는다.

상태:

- 완료: central supervised/Query SSL entrypoint는 이미 thin wrapper라 유지했다.
- 완료: FSSL entrypoint는 `run_sweep_if_requested()`,
  `resolve_single_simulation_output_dir()`, `run_single_simulation_from_config()`,
  `print_simulation_result()`로 분리해 첫 화면에서 sweep/single-run 흐름이 보이게
  했다.

### 4단계: runtime preparation과 execution loop 분리

FSSL은 `simulation.py`, 중앙 SSL은 `scripts/support/query_ssl_text_encoder/runners/*`
를 중심으로 본다.

목표:

```text
prepare runtime
bootstrap/load inputs
run loop
write artifacts/report
```

각 단계가 함수 이름에서 드러나게 한다.

상태:

- 완료: FSSL `run_simulation_request()`를 runtime 준비, round loop 실행, result 조립
  흐름으로 분리했다.
- 완료: 중앙 Query SSL runner를 unlabeled view/context/algorithm 준비, 학습,
  평가, manifest 조립, artifact 저장 흐름으로 분리했다.
- 유지: 중앙 supervised runner는 이미 context 준비, 학습, 평가, artifact 저장 순서가
  드러나므로 추가 분리를 하지 않았다.

### 5단계: FSSL round lifecycle phase 정리

이 단계는 FSSL 전용으로 남기되, 전체 계획 안의 하위 단계로 취급한다.

대상:

```text
scripts/experiments/fl_ssl/federated_simulation/flow/round_loop.py
```

목표 phase:

```text
server step
open round
select clients
prepare peer context
train clients
build sync state
finalize publication
evaluate validation
cleanup transient resources
assemble summary
```

주의:

- `round_timing_breakdown` key 의미를 바꾸지 않는다.
- cache cleanup 같은 안정화 변경과 구조 리팩터링을 같은 커밋에 섞지 않는다.

### 6단계: README/index 갱신

코드 구조가 정리된 뒤에만 문서를 갱신한다.

후보:

- `scripts/experiments/central/ssl_control/README.md`
- `scripts/experiments/fl_ssl/README.md`
- `docs/execution_index.md`

목표는 cookbook 확대가 아니라, 처음 읽을 파일 순서를 짧게 남기는 것이다.

## 진행 원칙

- 한 번에 한 entrypoint 또는 한 execution phase만 다룬다.
- 기능 변경과 readability refactor를 섞지 않는다.
- 새 abstraction은 두 실행 레일 이상에서 같은 의미가 확인된 뒤에만 만든다.
- 단일 사용처 helper, 얇은 facade, 이름만 있는 compatibility layer는 만들지 않는다.
- 각 단계는 테스트 또는 smoke로 닫는다.
