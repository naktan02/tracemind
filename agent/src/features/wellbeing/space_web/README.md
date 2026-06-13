# Wellbeing Space Web

`space_web/`는 가족용 확장 프로그램의 아이용 분석 화면에 전달할
카테고리 노드/edge view-model을 만든다.

경계:

- 입력 source of truth는 agent-local `AnalysisEvent.category_scores`다.
- 출력 source of truth는 `WellbeingSpaceWebPayload`다.
- UI는 `nodes`와 `edges`만 그리며 baseline, delta, edge weight 의미를 재해석하지 않는다.
- 알고리즘은 `SpaceWebProjectionStrategy` 구현으로 교체한다.

현재 기본 구현은 `coactivation_delta`다. 최근 기간 안에서 개인 baseline보다
같이 올라간 카테고리 쌍에 edge weight를 부여한다. 이후 transition, embedding
trajectory, personalized graph 구현이 필요하면 같은 strategy protocol 아래에
추가한다.
