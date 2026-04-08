# TraceMind Server Monorepo

TraceMind는 `personalized local inference + federated shared model improvement`
를 함께 다루는 monorepo다. 이 저장소는 단일 앱이 아니라 공용 계약,
agent 로컬 runtime, server orchestration, 실험층을 역할별로 분리해 둔다.

처음 읽을 때는 코드보다 문서 진입점을 먼저 보는 편이 빠르다.

1. [docs/execution_index.md](/home/jmgjmg102/tracemind_server/docs/execution_index.md)
2. [plan.md](/home/jmgjmg102/tracemind_server/plan.md)
3. [docs/project_execution_plan.md](/home/jmgjmg102/tracemind_server/docs/project_execution_plan.md)
4. [shared/src/contracts/README.md](/home/jmgjmg102/tracemind_server/shared/src/contracts/README.md)

## Top-Level Ownership

- `shared/`
  - agent, main_server, scripts가 함께 읽는 contract와 canonical rule
- `agent/`
  - 로컬 추론, training example preparation, local training runtime
- `main_server/`
  - FL round lifecycle, aggregation, publication orchestration
- `scripts/`
  - synthetic simulation, sweep, report, visualization 같은 실험층
- `tests/`
  - cross-boundary integration/e2e

## Code Reading Quick Start

- agent 경로: [agent/src/services/README.md](/home/jmgjmg102/tracemind_server/agent/src/services/README.md)
- main_server 경로: [main_server/src/services/README.md](/home/jmgjmg102/tracemind_server/main_server/src/services/README.md)
- round orchestration 경로: [main_server/src/services/rounds/README.md](/home/jmgjmg102/tracemind_server/main_server/src/services/rounds/README.md)
- experiment 경로: [scripts/experiments/README.md](/home/jmgjmg102/tracemind_server/scripts/experiments/README.md)
