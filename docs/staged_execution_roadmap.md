# TraceMind Staged Execution Roadmap

이 문서는 phase 이름과 큰 순서만 빠르게 확인하는 축약본이다.
세부 설명과 현재 우선순위는
[`docs/project_execution_plan.md`](/home/jmgjmg102/tracemind_server/docs/project_execution_plan.md)
를 기준으로 본다.

## Phase Map

### Phase 0. 계약 정리

- 활성 contract와 보관 contract를 분리한다.

### Phase 1. 로컬 개인화 추론 MVP

- 입력 이벤트에서 개인화된 `AssessmentResult`를 만든다.

### Phase 2. 로컬 update 생성 MVP

- pseudo-label 또는 feedback 기반으로 local update를 생성한다.

### Phase 3. 중앙 FL coordinator MVP

- round/task/update/revision publication을 닫는다.

### Phase 4. end-to-end federation

- 여러 agent가 참여하는 루프를 재현한다.

### Phase 5. richer shared adapter

- diagonal scale보다 표현력 있는 shared adapter를 검토한다.

### Phase 6. privacy hardening

- clipping, secure aggregation, DP, 필요 시 HE를 추가한다.

## 확인 포인트

1. shared representation과 local personalization이 분리돼 있는가
2. training, aggregation, privacy가 섞여 있지 않은가
3. source of truth가 코드 가까이에 있는가
4. full encoder FL이 너무 이르게 열리지 않았는가
