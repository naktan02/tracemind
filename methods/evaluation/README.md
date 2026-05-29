# Evaluation

`methods/evaluation/`은 중앙 SSL과 FL SSL이 함께 쓰는 평가 metric 계산 helper를
소유한다.

이 패키지는 runner orchestration, artifact 저장, 논문 표 생성은 맡지 않는다. 각
실험 entrypoint는 여기서 만든 canonical metric payload를 자기 report schema에
포함한다.

- `classification_report.py`: 분류 평가 report의 canonical metric shape
- `classification_payload.py`: classification report를 track별 evaluation payload로
  옮길 때 쓰는 typed per-label/confusion-matrix 정규화
- `pseudo_label_quality.py`: selection result 또는 final pseudo-label snapshot을
  client/round report가 소비하는 accepted/correct/distribution diagnostic으로 요약
