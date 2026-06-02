# Dash

이 패키지는 USB `semilearn/algorithms/dash`의 핵심 train-step과
`DashThresholdingHook`을 TraceMind Query SSL algorithm seam에 맞춘 구현이다.

보존한 USB core:

- weak/strong consistency objective
- weak logits에서 만든 soft pseudo-label 기반 threshold mask
- `rho = max(C * gamma^-cnt * rho_init, rho_min)` dynamic threshold
- `rho_min` 도달 시 hard pseudo-label 전환
- `sup_loss + lambda_u * unsup_loss` 결합

TraceMind 차이:

- USB의 별도 supervised warm-up phase는 중앙 SSL의 supervised seed 시작점과 맞지
  않으므로, 학습 시작 전 selection loss를 `rho_init`으로 주입한다.
- Hydra/config, tokenizer, row 준비, artifact 저장은 scripts runner가 소유한다.
