# FedAvg Pseudo-Label

`fedavg_pseudo_label`은 현재 TraceMind FL SSL simulation의 활성 baseline
method descriptor다.

이 패키지는 method 이름과 조합 metadata만 소유한다.

- `descriptor.py`: method identity, required views, runtime capability, local/server
  hint를 소유한다.
- `fedavg_pseudo_label.py`: builtin registry wiring만 담당한다.

Hydra config loading, client training service 생성, round request 조립,
artifact/report 저장은 `scripts/experiments/fl_ssl/federated_simulation/`에
남긴다.
