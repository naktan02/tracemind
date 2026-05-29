# Query Text Views

이 패키지는 query-domain text row/view를 tokenizer batch로 바꾸는 입력 glue를
소유한다. frozen text backbone에 PEFT adapter와 classifier head를 얹는 재사용
scaffold의 source of truth는
`methods/adaptation/peft_text_encoder/`가 소유한다.

범위:

- `data.py`: labeled row와 weak/strong row를 tokenizer batch로 바꾸는 입력 glue
- `view_rows.py`: weak/original/strong view row 해석과 validation helper
  - training example backend 이름 자체는 `shared` contract가 소유하고, 이 파일은
    해당 backend가 요구하는 query row shape 검증만 맡는다.
- `unlabeled_preparation.py`: Query SSL descriptor가 요구하는 view surface에 맞춰
  unlabeled row를 검증하거나 strong candidate를 생성/재사용하는 준비 core
- `query_ssl_views.py`: 중앙 SSL과 FL client-local SSL이 공유하는 unlabeled
  view surface별 DataLoader builder
- `local_training_budget.py`: Query SSL local step/batch budget을 labeled/unlabeled
  loader 노출과 분리해 계산하는 helper
- `tokenization.py`: run-local text tokenization cache와 padding 정규화 helper

범위 밖:

- Query SSL objective 자체는 `methods/ssl/`이 소유한다.
- LoRA/RSLoRA 같은 PEFT adapter builder는
  `methods/adaptation/peft_adapters/`가 소유한다.
- PEFT + classifier 모델/학습 scaffold는
  `methods/adaptation/peft_text_encoder/`가 소유한다. 삭제된
  `methods/adaptation/lora_classifier/` direct import path는 repo 내부 compatibility
  표면이 아니다.
- prototype 기반 query adaptation이 추가되더라도 이 패키지는 text view와
  token-batch glue만 제공한다. prototype building/scoring/training 의미는
  `methods/prototype/**` 계층이 소유하고, 이 패키지는 tokenizer/view 입력만 넘긴다.
- agent API, local private state, query buffer repository 접근은 agent layer에
  남긴다.
- shared update payload, artifact materialization, PEFT model composition은 이
  패키지가 소유하지 않는다.
