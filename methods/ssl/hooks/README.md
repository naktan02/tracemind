# SSL Hooks

`methods/ssl/hooks/`는 중앙 SSL과 FL SSL client가 함께 재사용하는 SSL
subroutine을 둔다.

## 책임

- `pseudo_labeling.py`: weak-view prediction을 hard/soft pseudo-label target으로 변환
- `masking.py`: unlabeled consistency loss에 적용할 sample mask 생성
- `consistency.py`: strong-view logits와 weak-view target 사이의 masked consistency
  loss 계산
- `objective.py`: algorithm이 role별 hook을 명시적으로 받는 bundle 정의
- `selection.py`: `PseudoLabelEvidence`를 threshold 기반 selection decision으로 해석
- `acceptance.py`: `acceptance_policy_name`을 selection hook spec으로 해석하는
  method-level metadata
- `teacher.py`: offline 또는 checkpoint teacher가 unlabeled row에 prediction evidence를
  제공할 때 쓰는 provider hook 계약
- `registry.py`: scripts와 agent runtime이 공유하는 selection hook lookup
- builtin hook import trigger는 registry 내부의 bounded package import로 처리하고,
  registration은 각 hook 구현 옆 decorator가 소유

## 제외

- `PseudoLabelCandidate` 생성과 selection context/diagnostics 조립
- query buffer retention, local training artifact 저장
- FL round lifecycle, aggregation, publication

위 실행/진단 glue는 `agent/src/services/training/selection/`과 scripts runner에
남긴다.

## Algorithm Hook 조합 방식

USB/SemiLearn처럼 알고리즘별로 pseudo-labeling, masking, consistency loss를
교체할 수 있어야 한다. 다만 TraceMind는 mutable hook dict와 문자열 role lookup을
그대로 들여오지 않고, `SslObjectiveHooks` typed bundle로 role을 명시한다.

예를 들어 FixMatch는 `HardOrSoftPseudoLabelingHook`,
`FixedThresholdMaskingHook`, `CrossEntropyConsistencyLossHook` 조합을 사용한다.
FreeMatch처럼 thresholding만 다른 알고리즘은 처음에는
`methods/ssl/algorithms/freematch/thresholding.py` 같은 method-local hook으로 두고,
해당 hook을 `SslObjectiveHooks.masking`에 꽂는다.

두 개 이상 algorithm에서 안정적으로 공유되는 것이 확인될 때만 algorithm-local hook을
`methods/ssl/hooks/`로 승격한다. 이 규칙은 공통 SSL hook이 특정 algorithm의 상태나
편향을 먼저 흡수하지 않게 하기 위한 것이다.
