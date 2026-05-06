# LoRA Classifier Adaptation

`methods/adaptation/lora_classifier/`는 frozen text backbone 위에 LoRA/PEFT
adapter와 classifier head를 얹는 재사용 가능한 scaffold를 소유한다.

범위:

- `modeling.py`: transformer backbone 로드, PEFT adapter 적용, classifier head 조립
- `training.py`: supervised / query SSL 학습 루프와 평가

범위 밖:

- raw query row 로딩과 tokenizer batch 생성 glue는 현재
  `methods/adaptation/query_classifier_adaptation/data.py`에 남긴다.
- LoRA/RSLoRA adapter builder 자체는 `methods/adaptation/lora/`와
  `methods/adaptation/peft/`가 소유한다.
- FL runtime state/update payload, aggregation, publication은 이후
  `lora_classifier` family contract 단계에서 `shared`, `agent`, `main_server`에
  별도로 연다.
