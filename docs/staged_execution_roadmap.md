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
