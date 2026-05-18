# FedAvg Pseudo-Label

`fedavg_pseudo_label`은 현재 TraceMind FL SSL simulation의 활성 baseline
method descriptor다.

이 패키지는 method 이름, 조합 metadata, method-local policy hint만 소유한다.

- `descriptor.py`: method identity, required views, runtime capability, local/server
  hint와 현재 baseline의 지원 profile/backend/family recipe를 함께 소유한다.
- `local_objective.py`: client local pseudo-label objective 의미를 소유한다.
- `server_policy.py`: server aggregation policy hint를 소유한다.
- `round_policy.py`: round별 method policy hint를 소유한다.

별도 `recipe.py`는 조합표가 커져 `descriptor.py` 가독성을 해칠 때만 연다.
별도 registry wiring shim은 두지 않고, registry가 `descriptor.py`의 `descriptor`
변수를 convention으로 등록한다.

Hydra config loading, client training service 생성, round request 조립,
artifact/report 저장은 `scripts/experiments/fl_ssl/federated_simulation/`에
남긴다.
