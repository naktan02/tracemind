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
```

`methods`는 `shared`와 외부 ML 라이브러리만 import한다. `agent`,
`main_server`, `scripts`를 import하지 않는다.

## 하위 패키지 지도

- `methods/ssl/fixmatch/`: USB 스타일 FixMatch objective core
- `methods/ssl/hooks/`: 중앙/FL SSL이 공유하는 pseudo-labeling, masking,
  selection hook
- `methods/adaptation/peft/`: PEFT adapter builder seam
- `methods/adaptation/lora/`: LoRA/RSLoRA builder core
- `methods/adaptation/lora_classifier/`: frozen backbone + LoRA/PEFT adapter +
  classifier head scaffold와 학습/평가 loop
- `methods/adaptation/diagonal_scale/`: diagonal-scale heuristic update 계산
- `methods/federated/aggregation/fedavg/`: FedAvg 가중 평균과 adapter family별
  next-state 계산 core
- `methods/federated/shard_policy/`: FL non-IID client shard assignment 계산
- `methods/federated_ssl/`: FL SSL method descriptor와 조합 metadata
- `methods/prototype/building/`: prototype pack builder와 single/kmeans/dbscan
  생성 전략
- `methods/prototype/scoring/`: prototype similarity와 category score policy 계산
- `methods/prototype/evidence/`: prototype score를 pseudo-label evidence로 정규화
- `methods/prototype/training_inputs/`: prototype single/multiview input view 계산

구현 상태와 기본 선택값은 `docs/strategy_surface_map.md`와 `conf/README.md`를
기준으로 본다. 이 문서는 `methods/`의 책임 경계만 설명한다.
