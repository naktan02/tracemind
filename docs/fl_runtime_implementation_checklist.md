# FL Runtime Implementation Checklist

이 문서는 시스템 FL 트랙의 짧은 구현 체크리스트다. 전체 연구 순서는
`docs/project_execution_plan.md`, 현재 경계는 `docs/architecture/system-overview.md`를
source of truth로 본다.

## 목표

- `agent`는 local inference/training과 FL participant runtime을 소유한다.
- `main_server`는 round lifecycle, update ingestion, aggregation, publication을 소유한다.
- `scripts`는 synthetic simulation, benchmark, evaluation harness로만 남긴다.
- 알고리즘 계산 core는 `methods`, 계약은 `shared`에 둔다.

## 구현 표면

- main_server round 생성, 조회, update ingest, finalize, publication.
- agent round client와 local training service.
- adapter family 기반 aggregation wiring.
- methods-owned `fedavg` aggregation core와 main_server generic executor.
- prototype rebuild/publication path.
- FL simulation harness와 artifact/report dump.
- architecture guard로 `scripts -> agent/main_server` 직접 import 제한.

## 약한 경계

- classifier-first baseline의 live agent path 확장.
- FedMatch/FedLGMatch/(FL)^2 같은 FL SSL method 실제 구현.
- `lora_classifier` family의 FL simulation research path, runtime translation
  payload와 aggregation.
- learned scorer artifact lifecycle.
- secure aggregation/DP runtime.
- long-running multi-agent integration smoke 안정화.

## Phase 0. 결정 고정

- [x] server-owned canonical prototype rebuild input.
- [x] polling 기반 agent participation.
- [x] update idempotency/duplicate 정책.
- [x] manual finalize 기반 round close.
- [ ] secure aggregation/DP 도입 시점 결정.

## Phase 1. Main Server Runtime

- [x] round lifecycle service.
- [x] round repository.
- [x] FL round API.
- [x] update acceptance.
- [x] aggregation adapter.
- [x] model/prototype publication.

완료 기준: 서버만 띄워도 round open, update ingest, finalize, publication이 된다.

## Phase 2. Agent Runtime

- [x] active round/task fetch.
- [x] local training task execution.
- [x] pseudo-label selection.
- [x] update build/upload.
- [x] unsupported backend 조기 종료.
- [ ] classifier-head live path coverage 강화.

완료 기준: agent가 raw/private state를 서버로 보내지 않고 update payload만 업로드한다.

## Phase 3. Simulation Harness

- [x] synthetic client shard.
- [x] runtime core를 쓰는 FL SSL smoke.
- [x] artifact dump와 report.
- [x] method descriptor와 shard policy config.
- [x] seed sweep runner와 summary report.
- [ ] 논문 method 비교군 추가.

완료 기준: 같은 split/seed/budget에서 method별 report를 재현할 수 있다.

## Phase 4. FL SSL Main Comparison

- [ ] `10 clients`, Dirichlet `alpha=0.3`, `3 seeds`, `50 rounds` main run.
- [ ] Dirichlet `alpha=0.1` stress run.
- [x] `10% labeled / 90% unlabeled` client pool 고정.
- [ ] macro-F1, worst-client macro-F1, ECE, communication cost report.
- [ ] 중앙 SSL control table과 FL SSL ranking 분리.

## Phase 5. Runtime Translation

- [x] `lora_classifier` simulation family의 state/update shape를 먼저 smoke 검증.
- [ ] winner method가 요구하는 shared family/state/update payload 정의.
- [x] agent local trainer scaffold 구현. raw text는 agent-local 입력으로만 쓰고
  update payload에는 artifact ref와 통계만 남긴다.
- [ ] agent LoRA artifact upload/materialization 구현.
- [x] main_server aggregation/publication adapter scaffold 구현. inline delta FedAvg와
  server-owned `aggregation_artifact::` JSON artifact-ref update를 검증한다.
- [x] main_server LoRA artifact materializer/loader 1차 구현. 현재 범위는
  server-owned JSON artifact ref이며, `agent-local://` ref는 upload 경로가 붙기
  전까지 거부한다.
- [x] FL simulation에서 `lora_pseudo_label_v1` local profile과
  `fedavg_lora_classifier` round-runtime profile을 선택할 수 있게 연결한다.
- [ ] backward-compatible manifest/version 정책 확인.
- [ ] architecture guard와 integration smoke 추가.

## LoRA-classifier 검증 게이트

- [x] unit: payload parse/serialize, training algorithm profile, LoRA config snapshot.
- [x] unit: `LoraTextClassifier` 1-batch train/eval step.
- [x] unit: methods-owned LoRA-classifier FedAvg inline delta shape/version과
  server-owned artifact-ref materialization.
- [x] smoke: `hash_debug + cpu_local` baseline `2 clients / 1 round / 1 seed`.
- [x] small: `hash_debug + cpu_local` baseline `3 clients / 2 rounds / 1 seed`.
- [x] LoRA bootstrap: `lora_pseudo_label_v1 + fedavg_lora_classifier`
  `2 clients / 0 rounds / 1 seed`.
- [ ] LoRA smoke: `2 clients / 1 round / 1 seed`.
  methods-owned LoRA-classifier FedAvg strategy는 inline delta와 server-owned
  artifact-ref update를 집계한다. 실제 1-round smoke는 agent가 만든
  `agent-local://` LoRA artifact를 server-owned artifact ref로 upload/materialize하는
  경로가 붙은 뒤 실행한다.
- [ ] standard 전 runtime trace: GPU memory, update size, round time.

## 완료 기준

- raw text와 개인 해석 상태는 agent-local boundary에 남는다.
- server는 round, aggregation, publication만 소유한다.
- scripts 없이도 agent/main_server runtime contract가 설명된다.
- scripts simulation은 production core를 복사하지 않고 호출한다.
- 새 method 추가 위치가 `methods/federated_ssl/<method>/`, `conf`, 필요한 capability
  adapter, test로 분명히 나뉜다.
