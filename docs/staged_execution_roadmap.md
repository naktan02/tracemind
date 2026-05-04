# TraceMind Staged Execution Roadmap

이 문서는 phase 이름과 큰 순서만 빠르게 확인하는 축약본이다.
세부 설명과 현재 우선순위는
[`docs/project_execution_plan.md`](project_execution_plan.md)
를 기준으로 본다.

## Phase Map

### Phase 0. 계약 정리

- 활성 contract와 보관 contract를 분리한다.

### Phase 1. 중앙집중형 seed baseline

- `central fixed embedding + classifier`를 canonical seed로 만든다.

### Phase 2. query 적응 준비

- query 버퍼, threshold/policy selection, 소량 수동 라벨 개입 지점을 정한다.

### Phase 3. 중앙집중형 SSL control 비교

- 같은 초기 seed checkpoint와 accepted query-derived rows를 기준으로
  `LoRA + classifier` continual adaptation 위에서
  `supervised -> pseudo-label self-training -> FixMatch -> R-Drop -> MixText`를
  pooled/offline control table로 비교한다.

### Phase 4. FL SSL non-IID 메인 비교

- client non-IID split 위에서 `FedMatch`, `FedLGMatch`, `(FL)^2` 같은
  FL-specific SSL 방법론을 메인 논문 비교선으로 닫는다.
- main condition은 `10 clients`, Dirichlet `alpha=0.3`, `10% labeled / 90% unlabeled`, `3 seeds`다.
- stress condition은 같은 조건에서 Dirichlet `alpha=0.1`로 둔다.
- main budget은 `50 communication rounds`, `local_epochs=1`, `max_steps=50`이다.
- smoke budget은 실행 확인용 `3 rounds`다.
- primary metric은 `macro-F1 + worst-client macro-F1`이고,
  `ECE`, communication cost, per-client variance는 tie-breaker/risk 지표다.
- 현재 active method baseline은 `strategy_axes/fl/method_descriptor=fedavg_pseudo_label`이다.
- 추가 논문 method는 후보 확정 전까지 구현 파일을 만들지 않는다.

### Phase 5. 시스템 FL runtime translation

- FL SSL winner를 우선 `LoRA family + classifier` 후보로 runtime/privacy 제약에 맞게 옮긴다.

### Phase 6. richer shared adapter

- diagonal scale보다 표현력 있는 shared adapter를 검토한다.

### Phase 7. privacy hardening

- clipping, secure aggregation, DP, 필요 시 HE를 추가한다.

## 확인 포인트

1. seed baseline과 query-domain 적응이 같은 주장으로 섞이지 않았는가
2. shared representation과 local personalization이 분리돼 있는가
3. query 버퍼와 training/aggregation/privacy가 섞여 있지 않은가
4. source of truth가 코드 가까이에 있는가
5. full encoder FL이 너무 이르게 열리지 않았는가


## Current Checkpoint

- `Phase 1` 방향은 `central fixed embedding + classifier` seed로 확정했고,
  canonical artifact는 `clf_2026_04_11_143138`으로 고정했다.
- query-domain 적응은 바로 학습하지 않고, 먼저 `query buffer` 로컬 저장과 selection 준비를 닫는다.
- `QueryBuffer`는 `agent` 소유 local state이고, source of truth는 `docs/contracts/query_buffer_v1.md`다.
- `agent`에는 `QueryBufferRepository`와 inference append 경로가 들어갔다.
- 현재 local query buffer 최소 조회는 `count`, `get(query_id)`, `get_recent(limit)` 기준으로 닫혔다.
- query buffer와 scored event를 `PseudoLabelEvidence`로 연결하는 projection helper가 들어갔다.
- query buffer snapshot을 기존 acceptance policy로 평가하는 selection runner가 들어갔다.
- accepted pseudo-label candidate를 raw-text adaptation dataset으로 조립하는 경로가 들어갔다.
- adaptation dataset은 `source_row.query_id`를 single source of truth로 두고,
  canonical provenance는 typed field로 유지한다.
- query buffer retention / purge는 agent-local config로 분리했고 기본값은 `30일 + 최신 5000건 유지`다.
- adaptation dataset label 기본 정책은 `pseudo_label_only`이며 future manual override hook을 열어 두었다.
- adaptation dataset을 기존 `train_lora_classifier` JSONL 입력 shape로 export하는 scripts bridge가 들어갔다.
- adaptation dataset export는 JSONL/manifest와 함께 summary JSON도 남긴다.
- adaptation dataset은 baseline runner에 labeled row를 메모리에서 직접 넘기고,
  export JSONL은 trace/audit 산출물로만 남긴다.
- `execution_context/runtime_env=auto_local` 기준으로 query adaptation dataset에서 supervised `LoRA + classifier` smoke run 1회를 검증했다.
- query-buffer selection 결과를 family-agnostic summary/trace diagnostics shape로 정리하고,
  JSON/JSONL dump로 저장하는 helper가 들어갔다.
- single-view query adaptation dataset에서 weak/strong source row를 만드는
  multiview preparation service와 JSONL export helper가 들어갔다.
- multiview preparation은 augmentation recipe를 고정하지 않고 pluggable augmenter hook으로 연다.
- supervised baseline 학습 코어를 `agent/src/services/training/query_classifier_adaptation/`로 옮기고,
  scripts는 entrypoint/artifact layer로 얇게 정리했다.
- canonical supervised baseline entrypoint는 `scripts/experiments/train_lora_classifier.py`로,
  concrete helper는 `scripts/experiments/lora_classifier/runner.py`와
  `query_adaptation_runner.py` direct import 기준으로 정리했다.
- 첫 pseudo-label bootstrap entrypoint로
  `train_lora_bootstrap_classifier_teacher.py`를 추가했고,
  `fixed embedding + classifier` teacher가 unlabeled pool에 pseudo-label을 붙인 뒤
  `LoRA + classifier` student를 학습한다.
- `pseudo-label self-training` runner를 추가했다.
- 현재 helper는 첫 bootstrap 이후 같은-family loop에서
  seed labeled rows와 pseudo-labeled rows를 합쳐 실행하는 offline 경로를 가진다.
- 다만 central canonical 비교 규약은 `seed checkpoint 1회 생성 -> 이후 new accepted query-derived rows only continual adaptation`으로 정리한다.
- 중앙 SSL 비교는 pooled/offline control이고, `FedMatch`, `FedLGMatch`, `(FL)^2`는
  FL SSL non-IID 메인 비교선으로 둔다.
- FL SSL main comparison 조건은 `10 clients`, Dirichlet `alpha=0.3`, `10/90 labeled/unlabeled`,
  `3 seeds`로 고정했고, `alpha=0.1`은 stress split으로 둔다.
- FL SSL main budget은 `50 communication rounds`, `local_epochs=1`, `max_steps=50`으로 고정했다.
- winner 1차 기준은 `macro-F1 + worst-client macro-F1`이며,
  `ECE`, communication cost, per-client variance는 tie-breaker/risk 지표다.
- FL SSL simulation report는 `fl_ssl_main_comparison` track으로 저장해 중앙 SSL control table과 분리한다.
- FL SSL method 선택 축은 열었고, 현재는 `fedavg_pseudo_label` baseline만 active runtime이다.
- selection 결과는 새 shape를 만들지 않고 기존 `PseudoLabelEvidence`, `PseudoLabelCandidate`, `DecisionFeedbackSignal`로 연결한다.
- 아직 하지 않는 것:
  - `lora family` shared/FL contract 추가
  - FL update 생성 및 서버 집계

## Next Session Checklist

1. `strategy_axes/fl/shard_policy=dirichlet_alpha03`와 `strategy_axes/fl/method_descriptor=fedavg_pseudo_label`
   smoke 실행으로 report JSON을 확인한다.
2. 구현 전 후보 FL SSL 논문 method를 비교하고, 실제 구현할 method를 확정한다.
3. 확정 method부터 `agent`/`main_server` 소유 경계에 구현한다.
4. `alpha=0.3` main과 `alpha=0.1` stress를 3 seeds로 실행할 sweep wrapper를 만든다.

## Guardrails

- `QueryBuffer` 단계에서는 저장만 하고 실제 학습은 하지 않는다.
- raw query text는 로컬에만 남기고 서버로 보내지 않는다.
- `ScoredEventRepository`와 `QueryBuffer` 역할을 섞지 않는다.
- `shared` contract에 아직 `lora family` payload를 추가하지 않는다.
