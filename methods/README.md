# Methods

`methods/`는 TraceMind 실험과 production adapter가 함께 재사용할 수 있는
알고리즘 core 패키지다.

## 책임

- SSL objective, pseudo-label, loss, thresholding 같은 method 계산
- PEFT/adaptation 적용 방식
- adapter family별 local update 계산 방식
- shared adapter scoring과 privacy guard 계산 방식
- federated aggregation의 순수 계산 core
- FL-SSL composition에서 재사용되는 method 조립 규칙
- FedProx 같은 adapter-family 중립 local objective regularizer 계산
- FL SSL method별 local objective, server policy, round policy 의미
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

새 알고리즘이나 논문 method를 추가할 때는 먼저 `methods/`에 method-local module을
만든다. `agent`와 `main_server`에 method 이름을 가진 runtime 파일을 늘리는 방식은
framework seam이 얕다는 신호로 본다. 두 runtime 계층은 raw text 접근, private
state, artifact repository, round lifecycle, publication처럼 실행 환경 차이를
core interface로 변환하는 adapter만 소유한다.

논문 방법론은 `methods/federated_ssl/<method>/`를 사람이 읽는 시작점으로 둔다. 이
폴더는 descriptor, recipe, local objective, server/round policy, method-only
aggregation 변형을 묶는다. 반대로 두 개 이상 방법론에서 재사용되는 SSL hook,
aggregation backend, adapter projection, prototype builder는 축별 패키지로 승격한다.
이 기준은 method 응집도와 교체 가능한 core 재사용성을 동시에 지키기 위한 것이다.
새 방법론이 기존 algorithm file 하나로 표현되지 않고 view, pseudo-label 생성,
confidence/weighting, consistency loss, distribution alignment, local regularizer,
teacher/memory, mix/adversarial, FL round-state 같은 새 변화 축을 요구하면 해당
방법론 추가 작업에 축 분리를 함께 포함한다. 기존 training loop나 runtime adapter에
method-specific 분기를 누적하지 않는다.
adapter family가 method-owned objective를 실행해야 할 때도
`methods/adaptation/<family>/federated_ssl/<method>_*.py`를 기본값으로 만들지
않는다. family 폴더는 `partitioned_training_loop.py` 같은 실행 primitive를 소유하고,
method 이름과 policy 의미는 descriptor와 `methods/federated_ssl/<method>/`에서 읽힌다.

## 하위 패키지 지도

- `methods/ssl/algorithms/fixmatch/`: USB 스타일 FixMatch objective core
- `methods/ssl/algorithms/freematch/`: USB 스타일 FreeMatch adaptive threshold
  objective core
- `methods/ssl/algorithms/adamatch/`: USB 스타일 AdaMatch distribution alignment
  objective core
- `methods/ssl/algorithms/pseudolabel/`: USB 스타일 PseudoLabel objective core
- `methods/ssl/hooks/`: 중앙/FL SSL이 공유하는 pseudo-labeling, masking,
  selection hook
- `methods/adaptation/peft_adapters/`: LoRA/DoRA 같은 PEFT mechanism builder와
  registry
- `methods/adaptation/local_objective_regularizers/`: FedProx처럼 adapter family와
  분리된 client-local objective regularizer
- `methods/adaptation/lora_classifier/`: 기존 `adapter_kind=lora_classifier`
  contract와 direct import 호환성을 위한 legacy shim package
- `methods/adaptation/diagonal_scale/`: diagonal-scale heuristic update 계산과
  family별 aggregation adapter
- `methods/adaptation/classification/`: modality-independent classification
  adapter primitive와 classifier-head projection
- `methods/adaptation/text_classifier/`: PEFT text encoder + classifier head
  adaptation variant와 text-specific training/update core
- `methods/adaptation/privacy_guards/`: shared adapter update clipping/DP
  policy core와 registry
- `methods/evaluation/`: 중앙 SSL과 FL SSL이 공유하는 평가 metric 계산 helper
- `methods/federated/aggregation/fedavg/`: FedAvg 공통 가중 평균 산술과 generic
  strategy wiring
- `methods/federated/shard_policy/`: FL non-IID client shard assignment 계산
- `methods/federated_ssl/`: FL SSL method descriptor, recipe, local objective,
  server/round policy, method-local variant
- `methods/prototype/building/`: prototype pack builder와 single/kmeans/dbscan
  생성 전략
- `methods/prototype/scoring/`: prototype similarity와 category score policy 계산
- `methods/prototype/evidence/`: prototype score를 pseudo-label evidence로 정규화
- `methods/prototype/training_inputs/`: prototype single/multiview input view 계산

구현 상태와 기본 선택값은 `docs/strategy_surface_map.md`와 `conf/README.md`를
기준으로 본다. 이 문서는 `methods/`의 책임 경계만 설명한다.
