# TraceMind Staged Execution Roadmap

이 문서는 phase 이름과 큰 순서만 빠르게 확인하는 축약본이다.
세부 설명과 현재 우선순위는
[`docs/project_execution_plan.md`](/home/jmgjmg102/tracemind_server/docs/project_execution_plan.md)
를 기준으로 본다.

## Phase Map

### Phase 0. 계약 정리

- 활성 contract와 보관 contract를 분리한다.

### Phase 1. 중앙집중형 seed baseline

- `central fixed embedding + classifier`를 canonical seed로 만든다.

### Phase 2. query 적응 준비

- query 버퍼, threshold/policy selection, 소량 수동 라벨 개입 지점을 정한다.

### Phase 3. 중앙집중형 적응 비교

- `LoRA + classifier` 적응 위에서 `FixMatch -> FreeMatch -> PabLO`를 비교한다.

### Phase 4. 시스템 FL baseline

- `fixed embedding + classifier_head` 기준 FL baseline을 닫는다.

### Phase 5. 시스템 FL translation

- 적응 winner를 우선 `LoRA family + classifier` 후보로 FL/runtime 제약에 맞게 옮긴다.

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

- `Phase 1` 방향은 `central fixed embedding + classifier` seed로 확정했다.
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
- adaptation dataset export 결과로 기존 `run_supervised_lora_baseline`을 호출하는 scripts 실행 helper가 들어갔다.
- `runtime=auto_local` 기준으로 query adaptation dataset에서 supervised `LoRA + classifier` smoke run 1회를 검증했다.
- selection 결과는 새 shape를 만들지 않고 기존 `PseudoLabelEvidence`, `PseudoLabelCandidate`, `DecisionFeedbackSignal`로 연결한다.
- 아직 하지 않는 것:
  - multiview/weak-strong query adaptation row 준비
  - `lora family` shared/FL contract 추가
  - FL update 생성 및 서버 집계

## Next Session Checklist

1. query-buffer selection summary / trace dump를 추가한다.
2. weak/strong augmentation이 필요한 family의 multiview row 준비 경로를 연다.
3. lifecycle purge를 어느 runtime cadence에서 실행할지 wiring한다.
4. supervised LoRA smoke 결과를 기준으로 weak-view evidence / acceptance policy 확장 지점을 고정한다.

## Guardrails

- `QueryBuffer` 단계에서는 저장만 하고 실제 학습은 하지 않는다.
- raw query text는 로컬에만 남기고 서버로 보내지 않는다.
- `ScoredEventRepository`와 `QueryBuffer` 역할을 섞지 않는다.
- `shared` contract에 아직 `lora family` payload를 추가하지 않는다.
