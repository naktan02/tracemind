# Methods

`methods/`는 TraceMind 실험과 production adapter가 함께 재사용할 수 있는
알고리즘 core 패키지다.

## 책임

- SSL objective, pseudo-label, loss, thresholding 같은 method 계산
- PEFT/adaptation 적용 방식
- federated aggregation의 순수 계산 core
- FL-SSL composition에서 재사용되는 method 조립 규칙
- prototype builder, assignment, update 같은 연구/production 공통 mechanism

## 제외

- Hydra entrypoint와 artifact 저장 로직
- FastAPI router, repository, HTTP transport
- main_server round lifecycle, update acceptance, publication
- agent-local private state, raw text, runtime storage
- 논문 표/그림/report 생성

## 의존 방향

```text
shared
  ↑
methods
  ↑
agent / main_server / scripts
  ↑
research
```

`methods`는 `shared`와 외부 ML 라이브러리만 import한다. `agent`,
`main_server`, `scripts`, `research`를 import하지 않는다.

## 진행 상태

현재 활성 구현:

- `methods/ssl/fixmatch/`: USB 스타일 FixMatch objective core
- `methods/adaptation/peft/`: PEFT adapter builder seam
- `methods/adaptation/lora/`: LoRA/RSLoRA builder core

다음 단계부터 FedAvg 계산 core, prototype method를 작은 단위로 옮긴다.
