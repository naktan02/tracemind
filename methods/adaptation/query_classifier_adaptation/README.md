# Query Classifier Adaptation

이 패키지는 query-domain classifier adaptation method scaffold의 token-batch 입력
glue를 소유한다. frozen text backbone에 PEFT adapter와 classifier head를 얹는
재사용 scaffold의 source of truth는
`methods/adaptation/text_classifier/peft_encoder/`가 소유한다.

범위:

- `data.py`: labeled row와 weak/strong row를 tokenizer batch로 바꾸는 입력 glue
- `view_rows.py`: weak/original/strong view row 해석과 validation helper
- `query_ssl_views.py`: 중앙 SSL과 FL client-local SSL이 공유하는 unlabeled
  view surface별 DataLoader builder
- `local_training_budget.py`: Query SSL local step/batch budget을 labeled/unlabeled
  loader 노출과 분리해 계산하는 helper
- `tokenization.py`: run-local text tokenization cache와 padding 정규화 helper

범위 밖:

- Query SSL objective 자체는 `methods/ssl/`이 소유한다.
- LoRA/RSLoRA 같은 PEFT adapter builder는
  `methods/adaptation/peft_adapters/`가 소유한다.
- LoRA + classifier 모델/학습 scaffold는
  `methods/adaptation/text_classifier/peft_encoder/`가 소유한다. 기존
  `methods/adaptation/lora_classifier/`는 shared contract v2 전까지 유지하는
  compatibility shim이다.
- prototype 기반 query adaptation이 추가되면 이 패키지 아래에 넣지 않는다.
  같은 token-batch classifier scaffold를 학습하는 경우에만 이 패키지를 재사용한다.
- agent API, local private state, query buffer repository 접근은 agent layer에
  남긴다.
- shared update payload, artifact materialization, PEFT model composition은 이
  패키지가 소유하지 않는다.
