# Query Classifier Adaptation

이 패키지는 query-domain classifier adaptation method scaffold의 데이터 로더,
모델 조립, 학습 루프를 소유한다. 현재 구현은 frozen text backbone에 PEFT adapter와
classifier head를 얹지만, PEFT adapter 자체는 이 패키지의 소유물이 아니다.

범위:

- `data.py`: labeled row와 weak/strong row를 tokenizer batch로 바꾸는 입력 glue
- `modeling.py`: frozen backbone, 외부 PEFT adapter builder, classifier head 조립
- `training.py`: supervised / Query SSL 학습 루프와 평가

범위 밖:

- Query SSL objective 자체는 `methods/ssl/`이
  소유한다.
- LoRA/RSLoRA 같은 PEFT adapter builder는
  `methods/adaptation/`이 소유한다.
- prototype 기반 query adaptation이 추가되면 이 패키지 아래에 넣지 않는다.
  같은 token-batch classifier scaffold를 학습하는 경우에만 이 패키지를 재사용한다.
- agent API, local private state, query buffer repository 접근은 agent layer에
  남긴다.
