# Methods

`methods/`는 TraceMind 실험과 production adapter가 함께 재사용할 수 있는
알고리즘 core 패키지다.

## 책임

- SSL objective, pseudo-label, loss, thresholding 같은 method 계산
- PEFT/adaptation 적용 방식
- adapter family별 local update 계산 방식
- federated aggregation의 순수 계산 core
- FL-SSL composition에서 재사용되는 method 조립 규칙
- prototype scoring/evidence 같은 local inference/training 공통 mechanism
- prototype 기반 학습 input view 계산

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
- `methods/ssl/hooks/`: 중앙/FL SSL이 공유하는 pseudo-labeling, masking,
  selection hook
- `methods/adaptation/peft/`: PEFT adapter builder seam
- `methods/adaptation/lora/`: LoRA/RSLoRA builder core
- `methods/adaptation/diagonal_scale/`: diagonal-scale heuristic update 계산
- `methods/federated/aggregation/fedavg/`: FedAvg 가중 평균과 adapter family별
  next-state 계산 core
- `methods/federated/shard_policy/`: FL non-IID client shard assignment 계산
- `methods/federated_ssl/`: FL SSL method descriptor와 조합 metadata
- `methods/prototype/scoring/`: prototype similarity와 category score policy 계산
- `methods/prototype/evidence/`: prototype score를 pseudo-label evidence로 정규화
- `methods/prototype/training_inputs/`: prototype single/multiview input view 계산

Prototype pack 생성 전략은 현재
`shared/src/services/prototypes/build_strategies.py`의 단일 표면을 쓴다.
single/kmeans/dbscan처럼 pack contract와 publication runtime이 함께 소비하는
builder는 별도 methods 표면을 만들지 않는다. 같은 알고리즘 표면을 두 군데로
쪼개야 할 만큼 change axis가 커질 때만 재검토한다.
