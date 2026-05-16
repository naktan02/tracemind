# LoRA Classifier Adaptation

`methods/adaptation/lora_classifier/`는 frozen text backbone 위에 LoRA/PEFT
adapter와 classifier head를 얹는 재사용 가능한 scaffold를 소유한다.

범위:

- `modeling.py`: transformer backbone 로드, PEFT adapter 적용, classifier head 조립
- `training.py`: supervised / query SSL 학습 루프와 평가
- `local_update.py`: FL agent runtime이 전달한 raw-text row, label schema,
  train executor artifact snapshot을 shared LoRA-classifier delta payload 의미로
  변환하는 local update core
- `training_backend.py`: `lora_classifier_trainer` local update backend registration
- `config.py`, `row_extractor.py`, `payload_builder.py`: objective extras,
  raw-text accepted example, payload-only artifact ref 조립
- `aggregation/fedavg.py`: LoRA-classifier delta를 FedAvg 공통 산술로 평균하고,
  shared state/update payload를 family FedAvg core 입력으로 변환한다. artifact IO나
  next-state payload 조립은 직접 소유하지 않는다.
- `aggregation/materialization.py`: inline delta와 server-owned JSON artifact ref를
  FedAvg 산술이 소비할 수 있는 delta mapping으로 읽는다. 기존 server state artifact도
  누적 global parameter snapshot으로 읽는다.
- `aggregation/state_projection.py`: `base global state + aggregated delta`를 다음
  LoRA/classifier global parameter snapshot과 `LoraClassifierState`로 투영한다.
- `server_update_materialization.py`: 서버가 읽을 수 없는 `agent-local://` delta
  artifact ref만 가진 update를 수락 전에 거부하는 family-local preflight
- FL simulation은 `inline_delta` 형식으로 서버가 직접 집계할 수 있는
  deterministic LoRA/classifier delta를 만들 수 있다. 이 경로는 lifecycle/FedAvg
  계약 검증용이며 실제 PEFT optimizer 산출물은 아니다.

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
- client update artifact는 base revision 기준 delta다. server state artifact는 delta가
  아니라 다음 라운드의 base가 되는 누적 global snapshot이다.
- `main_server`는 artifact ref 생성, 저장, publication만 담당한다. LoRA parameter나
  classifier-head payload의 의미 해석은 이 package가 소유한다.
- 새 adapter family도 같은 경계를 따른다. artifact materialization과 state projection이
  동시에 필요하면 family package 안에 같은 책임명으로 분리하고, runtime folder에
  method/family-specific 파일을 추가하지 않는다.
