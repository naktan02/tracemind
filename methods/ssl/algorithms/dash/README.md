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

- USB의 supervised warm-up phase를 trainer lifecycle에서 수행한 뒤 selection loss를
  `rho_init`으로 주입한다. 원본 기본 2048 step보다 낮춘 `num_wu_iter=1024`를
  TraceMind 중앙 PEFT 기본값으로 사용하며, smoke run은 Hydra override로 줄인다.
- Hydra/config, tokenizer, row 준비, artifact 저장은 scripts runner가 소유한다.
