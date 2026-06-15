# Fixed Feature Classification

`methods/classification/fixed_feature/`는 고정 feature 위에 얕은 classifier를 얹는
지도학습 baseline core를 소유한다.

## 책임

- 텍스트 row를 고정 feature로 변환하는 feature space 생성
- scikit-learn estimator 생성
- 학습, 예측, 분류 metric 계산

1차 feature space는 `tfidf_word`다. 이는 word unigram/bigram TF-IDF feature를 만들고,
encoder/backbone을 학습하지 않는다.

## 제외

- JSONL 파일 선택, Hydra 조합, artifact 저장 위치
- PEFT/full text encoder fine-tuning
- SSL objective와 unlabeled consistency loss

실행 orchestration과 report/artifact 저장은
`scripts/experiments/central/fixed_feature_control/`이 맡는다.
