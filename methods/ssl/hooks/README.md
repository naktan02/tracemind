# SSL Hooks

`methods/ssl/hooks/`는 중앙 SSL과 FL SSL client가 함께 재사용하는 SSL
subroutine을 둔다.

## 책임

- `pseudo_labeling.py`: weak-view prediction을 hard/soft pseudo-label target으로 변환
- `masking.py`: unlabeled consistency loss에 적용할 sample mask 생성
- `selection.py`: `PseudoLabelEvidence`를 threshold 기반 selection decision으로 해석
- `registry.py`: scripts와 agent runtime이 공유하는 selection hook lookup

## 제외

- `PseudoLabelCandidate` 생성과 selection context/diagnostics 조립
- agent-local `AcceptanceDecision` wrapping
- query buffer retention, local training artifact 저장
- FL round lifecycle, aggregation, publication

위 실행/진단 glue는 `agent/src/services/training/selection/`과 scripts runner에
남긴다.
