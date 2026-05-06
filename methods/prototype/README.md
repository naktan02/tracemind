# Prototype Methods

`methods/prototype/`는 prototype pack 생성과 prototype pack을 소비하는 method
mechanism을 둔다.

Prototype pack contract와 serialization은 `shared`가 소유한다. 어떤 알고리즘으로
prototype pack을 만들지는 `building/`이 소유한다.

Prototype 분석 실험 runner나 sweep은 `scripts/` 또는 `research/analysis/`가
소유한다.

## 하위 패키지 지도

- `building/`: single-centroid exact builder와 kmeans/dbscan multi-prototype
  build strategy
- `scoring/`: embedding-prototype similarity와 category score 집계 policy
- `evidence/`: prototype score나 score snapshot을 `PseudoLabelEvidence`로 정규화
- `training_inputs/`: prototype score 기반 single/multiview 학습 입력 view 계산
