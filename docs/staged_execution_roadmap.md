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
  `supervised -> USB PseudoLabel -> teacher pseudo-label self-training -> FixMatch -> R-Drop -> MixText`를
  pooled/offline control table로 비교한다.

### Phase 4. FL SSL non-IID 메인 비교

- client non-IID split 위에서 `FedMatch`, `FedLGMatch`, `(FL)^2` 같은
  FL-specific SSL 방법론을 메인 논문 비교선으로 닫는다.
- main condition은 `10 clients`, Dirichlet `alpha=0.3`, split `seed=42`,
  선택된 labeled/unlabeled source pool 전체 분배다.
- stress condition은 같은 조건에서 Dirichlet `alpha=0.1`로 둔다.
- main budget은 `50 communication rounds`, `local_epochs=1`, `max_steps=50`이다.
- smoke budget은 실행 확인용 `3 rounds`다.
- primary metric은 `macro-F1 + worst-client macro-F1`이고,
  `ECE`, communication cost, per-client variance는 tie-breaker/risk 지표다.
- method baseline과 추가 후보 구현 기준은 `docs/project_execution_plan.md`의
  active decision을 따른다.

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

## 현재 상태 위치

현재 checkpoint, next priority, 고정 artifact는 이 문서에 복제하지 않는다.
`docs/project_execution_plan.md`가 active decision과 next priority를 소유한다.

## Guardrails

- `QueryBuffer` 단계에서는 저장만 하고 실제 학습은 하지 않는다.
- raw query text는 로컬에만 남기고 서버로 보내지 않는다.
- `ScoredEventRepository`와 `QueryBuffer` 역할을 섞지 않는다.
- `shared` contract에 아직 `lora family` payload를 추가하지 않는다.
