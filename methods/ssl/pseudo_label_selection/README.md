# Pseudo-Label Selection

`methods/ssl/pseudo_label_selection/`은 `PseudoLabelEvidence`를 threshold 기반
selection decision으로 해석하는 순수 rule을 둔다.

## 책임

- `top1_margin_threshold`, `top1_confidence_only` 같은 selection algorithm
- confidence/margin threshold 적용
- scripts와 agent runtime이 공유하는 selection decision shape

## 제외

- `PseudoLabelCandidate` 생성과 selection context/diagnostics 조립
- agent-local `AcceptanceDecision` wrapping
- query buffer retention, local training artifact 저장

위 실행/진단 glue는 `agent/src/services/training/selection/`과 scripts runner에
남긴다.
