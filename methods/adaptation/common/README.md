# Adaptation 공통 scaffold

`methods/adaptation/common/`은 fixed embedding classifier와 LoRA classifier가
공유하는 classification scaffold만 둔다.

여기에는 model input 방식이나 runner orchestration을 넣지 않는다. feature tensor
모델과 token batch 모델은 각 adapter family가 계속 소유하고, 이 패키지는 평가
report 계산, best checkpoint 선택, selection-set epoch history record처럼
안정적으로 같은 부분만 담당한다.

- `classification_evaluation.py`: 평가 report의 canonical metric shape
- `checkpointing.py`: selection metric 기준 best model state 복원
- `training_history.py`: epoch history record와 logging summary
- `selection_training_loop.py`: caller가 소유한 epoch 학습 결과를 selection 평가,
  history, best checkpoint 흐름으로 닫는 얇은 scaffold
