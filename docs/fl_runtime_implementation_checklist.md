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
- `fedavg` aggregation core와 server adapter.
- prototype rebuild/publication path.
- FL simulation harness와 artifact/report dump.
- architecture guard로 `scripts -> agent/main_server` 직접 import 제한.

## 약한 경계

- classifier-first baseline의 live agent path 확장.
- FedMatch/FedLGMatch/(FL)^2 같은 FL SSL method 실제 구현.
- LoRA family의 FL runtime translation payload와 aggregation.
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
- [ ] 논문 method 비교군 추가.

완료 기준: 같은 split/seed/budget에서 method별 report를 재현할 수 있다.

## Phase 4. FL SSL Main Comparison

- [ ] `10 clients`, Dirichlet `alpha=0.3`, `3 seeds`, `50 rounds` main run.
- [ ] Dirichlet `alpha=0.1` stress run.
- [ ] `10% labeled / 90% unlabeled` client pool 고정.
- [ ] macro-F1, worst-client macro-F1, ECE, communication cost report.
- [ ] 중앙 SSL control table과 FL SSL ranking 분리.

## Phase 5. Runtime Translation

- [ ] winner method가 요구하는 shared family/state/update payload 정의.
- [ ] agent adapter 구현.
- [ ] main_server aggregation/publication adapter 구현.
- [ ] backward-compatible manifest/version 정책 확인.
- [ ] architecture guard와 integration smoke 추가.

## 완료 기준

- raw text와 개인 해석 상태는 agent-local boundary에 남는다.
- server는 round, aggregation, publication만 소유한다.
- scripts 없이도 agent/main_server runtime contract가 설명된다.
- scripts simulation은 production core를 복사하지 않고 호출한다.
- 새 method 추가 위치가 `methods`, `conf`, runtime adapter, test로 분명히 나뉜다.
