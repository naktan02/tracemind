# FedMatch

`fedmatch/`는 FedMatch 논문 method 의미를 소유한다.

- `descriptor.py`: method identity, required views, runtime capability surface
- `local_objective.py`: sigma/psi loss routing metadata
- `server_policy.py`: labels-at-client / labels-at-server policy metadata
- `round_policy.py`: helper context policy metadata
- `parameter_routing.py`: sigma/psi update partition metadata

v1 simulation은 공통 capability seam을 먼저 열어 둔다. 실제 LoRA-classifier
FedMatch loss loop와 server-side supervised step은 이 method package의 의미를
runtime adapter가 호출하는 다음 단계에서 구현한다.
