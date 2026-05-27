# LoRA Classifier Adaptation

`methods/adaptation/lora_classifier/`는 기존 shared contract와 runtime import 경로를
보존하는 legacy compatibility surface다. PEFT encoder + classifier head core의
source of truth는 `methods/adaptation/text_classifier/peft_encoder/`로 이동했다.

`config.py`, `evaluation.py`, `initial_state.py`, `runtime_compatibility.py`,
`server_preflight.py`, `training_backend.py`, `training/*`, `update/*`,
`aggregation/*`는 새 경로의 named symbol만 import하는 shim이다.
새 내부 코드는 `text_classifier/peft_encoder/`를 direct-file import로 참조한다.
기존 `adapter_kind=lora_classifier`와 payload format 이름은 shared contract v2
migration 전까지 유지한다.

## Compatibility Ledger

이 package에는 새 business logic을 추가하지 않는다. 남아 있는 `.py` 파일은 모두
`text_classifier/peft_encoder/`, `text_classifier/aggregation/`, 또는
`peft_adapters/`의 named symbol을 가져오는 direct-file compatibility shim이다.

유지 이유:

- shared contract 값이 아직 `adapter_kind=lora_classifier`와
  `lora_classifier_update`를 canonical runtime/artifact 이름으로 사용한다.
- 기존 FL simulation artifact, report verifier, notebook/script direct import가 이
  경로를 읽을 수 있다.
- contract v2 전까지 폴더명만 먼저 제거하면 producer/consumer drift가 생긴다.

제거 조건:

- `text_classifier_peft_*` 계열 shared contract v2 이름이 producer, consumer,
  verifier, report fixture에 동시에 반영된다.
- 기존 run artifact 호환 window가 끝나고 `docs/contracts/legacy_contract_ledger.md`의
  `lora_classifier` 항목이 닫힌다.
- architecture guard가 새 내부 import에서 legacy path 사용이 없음을 계속 보장한다.

기존 구현 범위는 다음과 같았다.

범위:

- `training/modeling.py`: transformer backbone 로드, PEFT adapter 적용,
  classifier head 조립
- `training/loops.py`: supervised / query SSL 학습 루프와 평가
- `training/step_budget.py`, `training/batching.py`, `training/optimizer_step.py`,
  `training/scalar_metrics.py`:
  supervised, query SSL, FL partitioned loop가 공유하는 step 분배, cycling batch,
  optimizer lifecycle, scalar metric 누적 primitive. loss/objective 의미,
  metric key prefix, partition 의미는 각 caller가 계속 소유한다.
- FedProx 같은 adapter-family 중립 local objective regularizer 계산은
  `methods/adaptation/local_objective_regularizers/`가 소유한다. 이 package의 loop는
  trainable parameter와 `proximal_mu`를 넘겨 적용 helper만 호출한다.
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
- `aggregation/fedavg.py`: 현재는
  `text_classifier/aggregation/peft_encoder_fedavg_projection.py` shim이다. 기존에는
  LoRA-classifier delta를 FedAvg 공통 산술로 평균하고,
  shared state/update payload를 family FedAvg core 입력으로 변환한다. artifact IO나
  next-state payload 조립은 직접 소유하지 않는다.
- `aggregation/partitioned_delta_average.py`: 현재는
  `text_classifier/aggregation/peft_encoder_partitioned_projection.py` shim이다. 기존에는
  shared update의 `partitioned_deltas`를
  client별로 먼저 병합한 뒤 LoRA-classifier delta 평균 산술과 state projection을
  재사용한다.
  partition 이름의 방법론 의미는 이 package가 아니라 method package가 소유한다.
- `aggregation/materialization.py`: 현재는
  `text_classifier/peft_encoder/update/materialization.py` shim이다. inline delta와
  server-owned JSON artifact ref를
  FedAvg 산술이 소비할 수 있는 delta mapping으로 읽는다. partitioned update도
  methods-owned partition delta object로 정규화한다. 기존 server state artifact도
  누적 global parameter snapshot으로 읽는다.
- `aggregation/state_projection.py`: 현재는
  `text_classifier/aggregation/peft_encoder_state_projection.py` shim이다.
  `base global state + aggregated delta`를 다음
  LoRA/classifier global parameter snapshot과 `LoraClassifierState`로 투영한다.
- `server_preflight.py`: update payload의 model/base revision/scope, backbone,
  LoRA config, label schema 호환성 및 서버가 읽을 수 없는 `agent-local://`
  artifact ref 거부 규칙
- FL simulation의 Query SSL LoRA-classifier 경로는 실제 PEFT optimizer 산출
  delta를 server-owned artifact ref로 저장하고 shared update에는 ref와 통계만
  남긴다. `inline_delta` deterministic executor는 legacy/debug lifecycle 검증용이다.

범위 밖:

- raw query row 로딩과 tokenizer batch 생성 glue는 현재
  `methods/adaptation/query_text_views/data.py`에 남긴다.
- agent-local query buffer 접근, payload 저장, upload, secure codec은 `agent`가
  소유한다.
- LoRA/RSLoRA adapter builder 자체는 `methods/adaptation/peft_adapters/`가
  소유한다.
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
