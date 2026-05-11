# AdaMatch

이 패키지는 USB `semilearn/algorithms/adamatch`의 핵심 train-step을
TraceMind Query SSL algorithm seam에 맞춘 구현이다.

보존한 USB core:

- weak/strong consistency objective
- labeled weak probability와 unlabeled weak probability 기반 distribution alignment
- labeled confidence 평균에 `p_cutoff`를 곱하는 relative confidence threshold
- aligned weak probability에서 pseudo-label을 만들고 strong logits에 CE consistency loss 적용

TraceMind 차이:

- USB의 hook dict/string lookup 대신 method-local hook instance를 직접 보유한다.
- `use_cat`, feature dict, checkpoint IO는 trainer/runtime 책임이라 algorithm core에서 제외한다.
- AdaMatch는 distribution alignment와 relative threshold에 labeled probability가 필요하므로
  `supervised_loss_weight=0.0`이어도 labeled batch를 요구한다.
