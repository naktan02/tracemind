# Methods

`methods/`는 TraceMind 실험과 production adapter가 함께 재사용할 수 있는
알고리즘 core 패키지다.

## 책임

- SSL objective, pseudo-label, loss, thresholding 같은 method 계산
- PEFT/adaptation 적용 방식
- payload/update-family별 local update 계산 방식
- shared adapter scoring과 privacy guard 계산 방식
- federated aggregation의 순수 계산 core
- FL-SSL composition에서 재사용되는 method 조립 규칙
- FedProx 같은 payload/update-family 중립 local objective regularizer 계산
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

Prototype 기반 classifier 대체가 추가되어도 `scripts`, `agent`, `main_server` 전반을
고치는 구조가 아니어야 한다. 새 scoring/building/update family는 `methods/`와
`conf/`에 추가하고, runtime 계층은 이미 합의된 capability contract만 호출한다.

최종 method/runtime 구조와 migration plan은
`docs/architecture/target-method-runtime-structure.md`를 우선한다. 현재 코드의
`adapter_family`, `lora_classifier` 이름은 legacy compatibility 표면을 설명할 수
있지만, 새 설계 판단에서는 `update_family`, `trainable_state`, `linear_head`,
`peft_text_encoder`, `prototype_pack` 용어를 기준으로 삼는다.

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
payload/update family가 method-owned objective를 실행해야 할 때도
`methods/adaptation/<family>/federated_ssl/<method>_*.py`를 기본값으로 만들지
않는다. family 폴더는 `partitioned_training_loop.py` 같은 실행 primitive를 소유하고,
method 이름과 policy 의미는 descriptor와 `methods/federated_ssl/<method>/`에서 읽힌다.

## 하위 패키지 지도

- `methods/ssl/algorithms/fixmatch/`: USB 스타일 FixMatch objective core
- `methods/ssl/algorithms/refixmatch/`: USB 스타일 ReFixMatch low-confidence KL
  objective core
- `methods/ssl/algorithms/freematch/`: USB 스타일 FreeMatch adaptive threshold
  objective core
- `methods/ssl/algorithms/adamatch/`: USB 스타일 AdaMatch distribution alignment
  objective core
- `methods/ssl/algorithms/dash/`: USB 스타일 Dash dynamic threshold objective core
- `methods/ssl/algorithms/simmatch/`: USB 스타일 SimMatch memory-bank similarity
  objective core
- `methods/ssl/algorithms/mixmatch/`: USB NLP 스타일 manifold MixMatch objective core
- `methods/ssl/algorithms/remixmatch/`: USB NLP 스타일 ReMixMatch DA + manifold
  MixUp objective core
- `methods/ssl/algorithms/pseudolabel/`: USB 스타일 PseudoLabel objective core
- `methods/ssl/hooks/`: 중앙/FL SSL이 공유하는 pseudo-labeling, masking,
  selection hook
- `methods/ssl/primitives/`: SSL algorithm들이 공유하는 probability transform,
  soft-target loss, MixUp, projection head 같은 순수 tensor/module primitive
- `methods/adaptation/peft_adapters/`: LoRA/DoRA 같은 PEFT mechanism builder와
  registry
- `methods/adaptation/local_objective_regularizers/`: FedProx처럼 payload/update family와
  분리된 client-local objective regularizer
- `methods/classification/linear_head/`: modality-independent linear classifier head
  primitive와 `classifier_head.v1(head_kind=linear)` payload projection. 새 MLP나
  projection head는 `linear_head` 아래에 넣지 않고 별도 classification owner를 연다
- `methods/adaptation/peft_text_encoder/`: PEFT text encoder + classifier head
  update family와 text-specific training/update core
- `methods/adaptation/query_text_views/`: query-domain text view row 해석,
  unlabeled view preparation, tokenizer batch glue
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
- `methods/prototype/projections.py`: prototype pack을 평가/리포트용 centroid view로
  투영하는 helper
- `methods/prototype/distance_report.py`: prototype centroid pairwise similarity/
  distance report 계산
- `methods/prototype/scoring/`: prototype similarity와 category score policy 계산
- `methods/prototype/evidence/`: prototype score를 pseudo-label evidence로 정규화
- `methods/prototype/training_inputs/`: prototype single/multiview input view 계산

구현 상태와 기본 선택값은 `docs/strategy_surface_map.md`와 `conf/README.md`를
기준으로 본다. 이 문서는 `methods/`의 책임 경계만 설명한다.
