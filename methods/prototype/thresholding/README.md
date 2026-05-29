# Prototype Thresholding

`methods/prototype/thresholding/`은 prototype score 위에서 동작하는 static threshold
policy core를 소유한다.

- `models.py`: threshold candidate 평가와 artifact 값 객체
- `evaluation.py`: scored prediction의 accepted set metric 계산
- `policies.py`: FixMatch식 fixed confidence, validation target error, classwise
  threshold policy 후보
- `selection.py`: validation metric 기반 candidate 선택/정렬 규칙

`scripts/experiments/prototype_analysis/`는 이 core를 실행하고 결과 JSON을 저장하는
runner/report 계층만 맡는다. 새 threshold policy를 추가할 때 기본 변경 위치는 이
폴더와 Hydra config이며, scripts runner에 policy 분기를 추가하지 않는다.
