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
- `fedavg.py`: LoRA-classifier delta를 FedAvg 공통 산술로 평균하는 family core
- `fedavg_projection.py`: LoRA-classifier state/update payload를 family FedAvg core
  입력으로 변환하고 logical artifact slot을 server context에 요청

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
