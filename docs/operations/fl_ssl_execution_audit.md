# FL SSL Execution Audit

작성일: 2026-05-18

이 문서는 현재 FL SSL 목표를 실제 artifact와 검증 명령 기준으로 대조한 감사표다.
장시간/full-budget 실험은 새로 실행하지 않고, 필요한 경우 1-round smoke와 기존
report/summary를 검증했다.

## 판정

전체 목표는 아직 완료가 아니다.

완료/검증된 항목:

- `FedAvg + FixMatch + LoRA-classifier` PEFT smoke와 artifact-ref delta 경로는
  구현 및 산출물 verifier로 확인됐다.
- `10 clients`, Dirichlet `alpha=0.3`, split seed `42`, `1 round`,
  `gpu_local + mxbai`, `FixMatch + FedAvg + LoRA-classifier` 현재 smoke report는
  verifier로 확인됐다.
- `agent-local://` LoRA/head delta를 server-owned `aggregation_artifact::` ref로
  upload/materialize하는 경로와 compatibility preflight는 구현 및 테스트로 닫혔다.
- `10 clients`, Dirichlet `alpha=0.3`, split seed `42`, `50 rounds`,
  `FixMatch + FedAvg + LoRA-classifier` main baseline report는 round/split/method/delta
  기준 verifier로 확인됐다. 이 report는 runtime metadata 도입 전 산출물이라
  `gpu_local + mxbai` 여부는 report 자체로는 재검증할 수 없다.
- `alpha=0.1` stress, FlexMatch/FreeMatch/PseudoLabel ablation은 5-round reduced
  run으로 method metadata와 실제 local objective 변경을 확인했다.
- `client_count=1..10` sweep은 1-round summary와 하위 report 10개를 확인했다.
- result index는 FL `fl_ssl_main_comparison.report.json`와 client-count sweep 하위
  report를 dashboard/index record로 정규화한다.
- FL SSL runner는 총 예정 communication round가 `49`를 넘으면 기본 차단한다.

사용자 승인 전까지 보류된 항목:

- Dirichlet `alpha=0.1` stress의 full `50 rounds` 실행.
- FlexMatch/FreeMatch/PseudoLabel full-budget ablation.
- `client_count=1..10` full sweep. `5 rounds x 10 client-counts`처럼 합계
  `50` rounds인 sweep도 long-run guard 대상이다.
- FedMatch/FedLGMatch/(FL)^2 중 첫 구현 method 선택.

## Prompt-To-Artifact Checklist

| 요구사항 | 현재 증거 | 상태 |
|---|---|---|
| 작업별 commit/push | 작업 단위 커밋은 `origin/master`에 push했다. 최신 상태는 `git status --short --branch`와 `git log --oneline origin/master -n 12`로 확인한다 | 완료 |
| scripts는 thin wrapper, method core는 methods 소유 | `scripts/experiments/fl_ssl/*`는 config/run/report, SSL algorithm은 `methods/ssl/algorithms/*`, LoRA aggregation은 `methods/adaptation/lora_classifier/*` | 완료 |
| 실제 PEFT 기준 FixMatch local objective 호출 | `tests/unit/test_lora_fixmatch_runner.py`, `tests/unit/test_run_federated_simulation.py`, current 1-round smoke report `objective.query_ssl.algorithm_name=fixmatch` | 완료 |
| LoRA/classifier delta가 FedAvg까지 집계 | `methods/adaptation/lora_classifier/aggregation/fedavg.py`, `main_server/tests/unit/test_aggregation_service.py`, artifact-ref verifier | 완료 |
| agent-local artifact upload | `scripts/runtime_adapters/federated_agent/query_ssl_lora_classifier_trainer.py`, `upload_agent_local_lora_classifier_update` | 완료 |
| server-owned materialization | `main_server/src/services/federation/rounds/aggregation/artifact_refs.py`, `methods/adaptation/lora_classifier/aggregation/materialization.py` | 완료 |
| manifest/version compatibility | `methods/adaptation/lora_classifier/server_update_compatibility.py`, `methods/adaptation/server_update_materialization.py`, related unit tests | 완료 |
| alpha=0.3 main baseline | `runs/federated_simulation/fixmatch_lora_alpha03_10c_50round_20260518/20260517T150549Z/reports/fl_ssl_main_comparison.report.json`는 round/split/method/delta 기준 PASS. runtime metadata는 current 1-round smoke와 reduced runs에서 검증 | 부분 |
| alpha=0.1 stress | 5-round reduced report verified. full 50-round는 사용자 지시로 보류 | 부분 |
| FlexMatch/FreeMatch/PseudoLabel ablation | 5-round reduced reports verified. full budget은 사용자 지시로 보류 | 부분 |
| client_count=1..10 sweep | 1-round summary verified and indexed. full sweep은 사용자 지시로 보류 | 부분 |
| seed sweep은 robustness로 분리 | `seed_sweep` runner/summary는 존재. split seed 42 안정화 뒤 별도 실행 | 대기 |
| 50-round 재실행 방지 | `scripts/experiments/fl_ssl/run_safety.py`와 `tests/unit/test_fl_ssl_run_safety.py` | 완료 |

## Read-Only Verification Evidence

아래 검증 중 첫 항목은 현재 코드 기준으로 새로 실행한 1-round smoke다. 나머지는
새 round를 실행하지 않고 기존 JSON artifact만 읽었다.

전체 검증 manifest:

```bash
uv run python scripts/experiments/fl_ssl/verify_federated_report_artifacts.py \
  --manifest docs/operations/fl_ssl_artifact_verification_manifest.current.json
```

결과: 7개 artifact entry 모두 `PASS`.

- `fixmatch_lora_alpha03_10c_1round_current_20260518` current smoke:
  `PASS runs/federated_simulation_smoke/fixmatch_lora_alpha03_10c_1round_current_20260518/20260517T232304Z/reports/fl_ssl_main_comparison.report.json`
  - 조건: `10 clients`, `alpha=0.3`, split seed `42`, `1 round`,
    `gpu_local + mxbai`, `FixMatch + FedAvg + LoRA-classifier`,
    `server_uploaded_artifact_ref`
  - 서버 update: `10`개, 모두 `aggregation_artifact::` ref이며
    `main_server/aggregation_artifacts/versions/lora_classifier/sim_rev_0001/`에
    누적 LoRA/head snapshot을 남겼다.
- `fixmatch_lora_alpha03_10c_50round_20260518` main:
  `PASS runs/federated_simulation/fixmatch_lora_alpha03_10c_50round_20260518/20260517T150549Z/reports/fl_ssl_main_comparison.report.json`
- `fixmatch_lora_alpha01_10c_5round_20260518` reduced stress:
  `PASS runs/federated_simulation_reduced/fixmatch_lora_alpha01_10c_5round_20260518/20260517T203139Z/reports/fl_ssl_main_comparison.report.json`
- `flexmatch_lora_alpha03_10c_5round_20260518` reduced ablation:
  `PASS runs/federated_simulation_reduced/flexmatch_lora_alpha03_10c_5round_20260518/20260517T205436Z/reports/fl_ssl_main_comparison.report.json`
- `freematch_lora_alpha03_10c_5round_20260518` reduced ablation:
  `PASS runs/federated_simulation_reduced/freematch_lora_alpha03_10c_5round_20260518/20260517T212002Z/reports/fl_ssl_main_comparison.report.json`
- `pseudolabel_lora_alpha03_10c_5round_20260518` reduced ablation:
  `PASS runs/federated_simulation_reduced/pseudolabel_lora_alpha03_10c_5round_20260518/20260517T214526Z/reports/fl_ssl_main_comparison.report.json`
- `fixmatch_lora_alpha03_1round_20260518` client-count sweep:
  `PASS runs/federated_simulation_client_count_sweep_short/fixmatch_lora_alpha03_1round_20260518/20260517T193320Z/reports/fl_ssl_client_count_sweep.summary.json`

Result index read-only ingest:

- `runs/federated_simulation_reduced` -> `indexed_runs=4`
- methods: `fixmatch_usb_v1`, `flexmatch_usb_v1`, `freematch_usb_v1`,
  `pseudolabel_usb_v1`
- algorithms: `fixmatch`, `flexmatch`, `freematch`, `pseudolabel`
- round budgets: `[5]`
- shard alphas: `[0.1, 0.3]`

## Next Gate

다음 실행성 작업은 사용자의 명시 승인이 필요하다. 승인 없이 진행 가능한 다음 작업은
FedMatch/FedLGMatch/(FL)^2 중 첫 method를 고르기 위한 capability matrix 작성과
method-owned descriptor/server-policy/round-state 요구사항 문서화다.
