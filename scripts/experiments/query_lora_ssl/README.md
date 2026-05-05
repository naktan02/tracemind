# Query LoRA SSL

이 패키지는 중앙 SSL control과 future FL client-local SSL에서 공유 가능한
query-domain LoRA SSL harness를 둔다. 직접 실행하는 Hydra entrypoint는
`scripts/experiments/central_ssl_control/*.py`가 소유하고, 이 패키지는 그
entrypoint가 호출하는 runner/helper 구현을 소유한다.

## 구조

- `runners/`
  - `supervised.py`: frozen backbone + LoRA + classifier supervised baseline.
  - `pseudo_label.py`: pseudo-label self-training wrapper.
  - `bootstrap_teacher.py`: fixed classifier teacher -> LoRA student bootstrap.
  - `consistency.py`: FixMatch 등 consistency-family Query SSL runner.
  - `query_adaptation.py`: agent-local query adaptation dataset runner wrapper.
- `harness/`
  - LoRA model/data/eval 공통 scaffolding.
- `io/`
  - run artifact 저장과 query adaptation dataset export.
- `config/`
  - Hydra initial checkpoint와 pseudo-label algorithm preset 해석.
- `query_ssl/`
  - Query SSL method 공통 모델과 augmentation preparation.

## 실행 경계

실험을 직접 실행할 때는 이 패키지의 runner 파일을 직접 실행하지 않고 아래
Hydra entrypoint를 실행한다.

- `scripts/experiments/central_ssl_control/train_lora_classifier.py`
- `scripts/experiments/central_ssl_control/train_lora_pseudo_label_classifier.py`
- `scripts/experiments/central_ssl_control/train_lora_fixmatch.py`
- `scripts/experiments/central_ssl_control/train_lora_bootstrap_classifier_teacher.py`

공통으로 재사용될 알고리즘 core는 이 패키지 안에서 키우지 않고 `methods/`로
올린다. cross-boundary contract/domain은 `shared/`, runtime bridge는
`scripts/runtime_adapters/`에 둔다.
