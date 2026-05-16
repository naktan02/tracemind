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
  state/update payload를 family FedAvg core 입력으로 변환한다. inline delta 또는
  server context가 제공한 artifact loader로 materialize한 delta를 처리하고,
  logical artifact slot을 server context에 요청한다. client update는 base revision
  기준 delta지만, server state artifact에는 `base global state + aggregated delta`
  결과인 누적 LoRA/classifier parameter snapshot을 저장한다.
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
