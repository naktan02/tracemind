# TraceMind Execution Index

## 목적

이 문서는 TraceMind 문서의 진입점이다.
작업 전에 어떤 문서를 먼저 읽어야 하는지, 각 문서가 무엇을 담당하는지 빠르게 확인하기 위한 용도다.

---

## 읽는 순서

1. [`plan.md`](/home/jmgjmg102/tracemind_server/plan.md)
   - 연구 비전, 문제 정의, 왜 이 구조를 택하는지

2. [`docs/project_execution_plan.md`](/home/jmgjmg102/tracemind_server/docs/project_execution_plan.md)
   - 구현 순서, 아키텍처 결정, MVP와 FL 확장 해석

3. [`docs/staged_execution_roadmap.md`](/home/jmgjmg102/tracemind_server/docs/staged_execution_roadmap.md)
   - 단계별 상세 실행 계획, 검증 기준, 사용자 확인 게이트

4. 계약 문서
   - [`docs/contracts/window_summary_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/window_summary_v1.md)
   - [`docs/contracts/prototype_pack_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/prototype_pack_v1.md)
   - [`docs/contracts/prototype_build_state_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/prototype_build_state_v1.md)
   - 이후 `norm_pack_v1.md`, `assessment_result_v1.md`가 추가되면 같은 레벨에서 참조

---

## 문서 역할

### 1. 비전 문서

- [`plan.md`](/home/jmgjmg102/tracemind_server/plan.md)
- 발표 메시지, 연구 목적, 장기 방향

### 2. 실행 계획 문서

- [`docs/project_execution_plan.md`](/home/jmgjmg102/tracemind_server/docs/project_execution_plan.md)
- 구현 순서와 구조적 결정

### 3. 실행 로드맵 문서

- [`docs/staged_execution_roadmap.md`](/home/jmgjmg102/tracemind_server/docs/staged_execution_roadmap.md)
- 실제 단계별 작업, 산출물, 검증, 사용자 확인 필요 지점

### 4. 계약 문서

- [`docs/contracts/window_summary_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/window_summary_v1.md)
- [`docs/contracts/prototype_pack_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/prototype_pack_v1.md)
- [`docs/contracts/prototype_build_state_v1.md`](/home/jmgjmg102/tracemind_server/docs/contracts/prototype_build_state_v1.md)
- 로컬/중앙 간 데이터 경계

### 5. 대화 기록

- [`docs/notes/README.md`](/home/jmgjmg102/tracemind_server/docs/notes/README.md)
- 세션 기록과 임시 메모

---

## 작업 시작 체크리스트

작업 전 최소 확인 순서:

1. `plan.md`에서 현재 연구 방향 확인
2. `docs/project_execution_plan.md`에서 현재 단계 확인
3. `docs/staged_execution_roadmap.md`에서 지금 작업이 어느 phase인지 확인
4. 관련 contract 문서 확인
5. 사용자 판단이 필요한 변경인지 확인

---

## 사용자 확인이 필요한 변경

아래 변경은 임의로 확정하지 않고 사용자 확인 후 진행한다.

1. cohort 분할 기준 변경
2. privacy 경계를 바꾸는 필드 추가
3. feedback/self-report 수집 도입
4. FL 단계 활성화
5. 주요 threshold 또는 decision policy 의미 변경
6. 사용자 노출 리소스 정책 변경
