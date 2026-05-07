# Federated SSL Methods

`methods/federated_ssl/`는 SSL local update와 federated aggregation을 조합하는
FL SSL method spec을 둔다. `fedavg_pseudo_label`도 예외 baseline이 아니라 이
spec을 만족하는 첫 method 구현으로 다룬다.

## 책임

- `federated_ssl_method` 이름별 method descriptor/spec
- required views, client local step, server round/aggregation 요구사항 표현
- simulation/live runtime capability 선언
- 실제 runtime adapter가 읽을 canonical method 요구사항 제공
- method-local module 옆 decorator 등록과 registry의 명시 builtin loader

## 제외

- Hydra config loading
- simulation loop와 artifact/report 저장
- `LocalTrainingService`, `RoundOpenRequest` 같은 runtime 객체 생성

위 실행 glue는 `scripts/experiments/fl_ssl/federated_simulation/`에 남긴다. 새
논문 method의 계산 core는 이 패키지나 `methods/ssl/*`, `methods/federated/*`
같은 `methods` 계층에 두고, `agent`/`main_server`에는 선택된 method의 runtime
adapter만 둔다.

새 method의 descriptor는 `methods/federated_ssl/<method>/descriptor.py`가
소유한다. `registry.py`는 자동 파일 스캔을 하지 않고, built-in method module을
명시적으로 import해서 decorator 등록을 실행한다.

새 method 추가 절차는 `NEW_METHOD.md`를 먼저 따른다.
