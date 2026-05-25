# LoRA Classifier Adaptation

`methods/adaptation/lora_classifier/`는 frozen text backbone 위에 LoRA/PEFT
adapter와 classifier head를 얹는 재사용 가능한 scaffold를 소유한다.

범위:

- `training/modeling.py`: transformer backbone 로드, PEFT adapter 적용,
  classifier head 조립
- `training/loops.py`: supervised / query SSL 학습 루프와 평가
- FedProx 같은 adapter-family 중립 local objective regularizer 계산은
  `methods/adaptation/local_objective_regularizers/`가 소유한다. 이 package의 loop는
  trainable parameter snapshot과 loss 추가 호출만 담당한다.
- `training/query_ssl_local_training.py`: FL Query SSL LoRA-classifier local
  training, 실제 PEFT optimizer 실행, delta 추출, shared update payload 조립 core.
  round 종료 시점의 local classifier pseudo-label snapshot quality도 계산한다.
- `federated_ssl/`: FL SSL method-local objective를 LoRA-classifier model/loaders,
  partitioned delta, update payload로 실행하는 adapter-family slice. method 의미는
  `methods/federated_ssl/<method>/`가 계속 소유하며, 이 폴더에는
  `<method>_*.py` 파일을 기본적으로 추가하지 않는다.
- `evaluation.py`: server-published global LoRA/classifier state를 materialize한 뒤
  중앙 SSL과 같은 classifier forward 평가를 실행하는 FL validation core
- `update/local_update.py`: FL agent runtime이 전달한 raw-text row, label schema,
  train executor artifact snapshot을 shared LoRA-classifier delta payload 의미로
  변환하는 local update core
- `update/payload_builder.py`: accepted example에서 raw-text row를 추출하고
  artifact ref 기반 update payload를 조립하는 adapter
- `update/query_ssl_update.py`: Query SSL 학습 결과를 shared update payload와
  client metric으로 조립
- `update/simulation_inline_delta.py`: deterministic inline-delta simulation executor
- `training_backend.py`: `lora_classifier_trainer` local update backend registration
- `config.py`: objective extras와 LoRA-classifier training/runtime payload config 해석
- `initial_state.py`: simulation/runtime bootstrap용 초기 shared LoRA-classifier
  state 생성 규칙
- `runtime_compatibility.py`: round runtime scaffold와 local objective payload
  config drift 검증
- `aggregation/fedavg.py`: LoRA-classifier delta를 FedAvg 공통 산술로 평균하고,
  shared state/update payload를 family FedAvg core 입력으로 변환한다. artifact IO나
  next-state payload 조립은 직접 소유하지 않는다.
- `aggregation/partitioned_delta_average.py`: shared update의 `partitioned_deltas`를
  client별로 먼저 병합한 뒤 LoRA-classifier delta 평균 산술과 state projection을
  재사용한다.
  partition 이름의 방법론 의미는 이 package가 아니라 method package가 소유한다.
- `aggregation/materialization.py`: inline delta와 server-owned JSON artifact ref를
  FedAvg 산술이 소비할 수 있는 delta mapping으로 읽는다. partitioned update도
  methods-owned partition delta object로 정규화한다. 기존 server state artifact도
  누적 global parameter snapshot으로 읽는다.
- `aggregation/state_projection.py`: `base global state + aggregated delta`를 다음
  LoRA/classifier global parameter snapshot과 `LoraClassifierState`로 투영한다.
- `server_preflight.py`: update payload의 model/base revision/scope, backbone,
  LoRA config, label schema 호환성 및 서버가 읽을 수 없는 `agent-local://`
  artifact ref 거부 규칙
- FL simulation의 Query SSL LoRA-classifier 경로는 실제 PEFT optimizer 산출
  delta를 server-owned artifact ref로 저장하고 shared update에는 ref와 통계만
  남긴다. `inline_delta` deterministic executor는 legacy/debug lifecycle 검증용이다.

범위 밖:

- raw query row 로딩과 tokenizer batch 생성 glue는 현재
  `methods/adaptation/query_classifier_adaptation/data.py`에 남긴다.
- agent-local query buffer 접근, payload 저장, upload, secure codec은 `agent`가
  소유한다.
- LoRA/RSLoRA adapter builder 자체는 `methods/adaptation/lora/`와
  `methods/adaptation/peft/`가 소유한다.
- FL runtime state/update payload shape는 `shared`, update upload와 artifact 저장은
  `agent`/`main_server`, generic aggregation method 산술/strategy wiring은
  `methods/federated/aggregation/`이 맡는다.

Aggregation package 규칙:

- `fedavg.py`는 집계 전략의 사람이 읽는 진입점이다. family 검증, FedAvg 산술 호출,
  strategy 등록만 남기고 artifact 읽기/쓰기 shape 조립을 넣지 않는다.
- `partitioned_delta_average.py`는 별도 method-specific runtime이 아니라 같은 family의
  server update 해석 backend다. `sigma/psi`처럼 왜 partition이 생겼는지는
  `methods/federated_ssl/<method>/`에서 읽는다.
- client update artifact는 base revision 기준 delta다. server state artifact는 delta가
  아니라 다음 라운드의 base가 되는 누적 global snapshot이다.
- `lora_classifier_eval`은 누적 global snapshot의 LoRA/head 파라미터를 실제
  classifier model에 로드해 평가한다. `prototype_similarity`는 prototype/selection
  계층용이며 LoRA/classifier 성능 validation으로 쓰지 않는다.
- `main_server`는 artifact ref 생성, 저장, publication만 담당한다. LoRA parameter나
  classifier-head payload의 의미 해석은 이 package가 소유한다.
- 새 adapter family도 같은 경계를 따른다. artifact materialization과 state projection이
  동시에 필요하면 family package 안에 같은 책임명으로 분리하고, runtime folder에
  method/family-specific 파일을 추가하지 않는다.
