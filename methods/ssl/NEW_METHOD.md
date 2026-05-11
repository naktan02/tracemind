# Query SSL New Method Guide

이 문서는 중앙 Query SSL control에 새 SSL objective를 추가할 때의 최소 경계를
고정한다. 대상은 `scripts/experiments/central_ssl_control/train_lora_query_ssl.py`
공통 entrypoint가 실행할 수 있는 `methods/ssl` algorithm이다.

중앙 SSL은 pooled/offline control이다. live agent training runtime이나
main_server round lifecycle을 통과하지 않는다.

## 먼저 분류한다

새 방법론을 구현하기 전에 아래 중 어디에 들어가는지 먼저 정한다.

| 종류 | 예시 | 난이도 | 기본 변경 위치 |
|---|---|---|---|
| weak-only step objective | PseudoLabel 변형 | 낮음 | `methods/ssl/algorithms/<method>/`, `conf/strategy_axes/ssl/consistency_method/` |
| weak/strong consistency objective | FixMatch, UDA, FreeMatch류 | 낮음-중간 | 위와 같음 |
| labeled/unlabeled batch 안에서 끝나는 regularization | R-Drop류 | 중간 | algorithm core + 필요 시 batch view validation |
| 새 unlabeled view surface가 필요한 objective | multi-strong view, custom augmentation | 중간 | algorithm core + dataloader/view builder + augmentation config |
| train phase가 다른 방법론 | TAPT, two-stage teacher/student | 중간-높음 | 기존 runner 재사용 여부를 먼저 판단, 필요 시 별도 phase runner |
| model/optimizer lifecycle을 바꾸는 방법론 | Mean Teacher EMA, adversarial training | 높음 | `QuerySslAlgorithm` seam 확장 또는 새 trainer capability |

현재 구조는 `compute_step(model, labeled_batch, unlabeled_batch)` 안에서 loss와 metric을
계산하는 방법론에 가장 잘 맞는다. 이 경계를 넘는 방법론을 억지로
`compute_step()`에 넣지 않는다.

## 기본 파일 구조

가장 작은 새 algorithm은 아래처럼 시작한다.

```text
methods/ssl/algorithms/<method_name>/
  __init__.py
  <method_name>.py
  README.md

conf/strategy_axes/ssl/consistency_method/<method_name>_v1.yaml
tests/unit/test_methods_<method_name>.py
```

`__init__.py`는 marker/docstring만 둔다. `registry.py`에 새 algorithm 목록을
직접 추가하지 않는다. `methods/ssl/registry.py`가
`methods/ssl/algorithms/<method>/<method>.py` convention으로 package module을
import하고, 해당 module의 decorator registration이 실행된다.

## 구현 순서

1. 고정 조건과 변경 변수를 적는다.
   - dataset/split, seed, backbone, tokenizer, LoRA config, initial checkpoint,
     metric, output metadata가 같은 비교표에서 고정되는지 확인한다.
2. 필요한 unlabeled view를 고른다.
   - weak-only면 `QuerySslRequiredViews(view_builder_name="usb_weak")`.
   - weak/strong이면 `USB_MULTIVIEW_REQUIRED_VIEWS`.
   - 새 view면 runner/dataloader 확장이 필요한지 먼저 판단한다.
3. method-local algorithm class를 만든다.
   - `algorithm_name`을 선언한다.
   - `uses_labeled_batches`를 정확히 반환한다.
   - `validate_loaders(...)`에서 빈 loader나 labeled batch 필요 여부를 검증한다.
   - `compute_step(...)`은 `QuerySslStepResult`를 반환한다.
4. 필요하면 lifecycle hook을 사용한다.
   - 전체 step 수가 필요하면 `configure_training(num_train_iter=...)`.
   - class 수나 unlabeled row 수가 필요하면 `configure_dataset(...)`.
   - 이 둘로 부족하면 먼저 seam 확장을 검토한다.
5. module 하단에서 `@register_query_ssl_algorithm(...)`로 factory를 등록한다.
6. Hydra config를 추가한다.
   - `name`: config preset 이름.
   - `algorithm_name`: registry 이름.
   - method hyperparameter와 `unlabeled_batch_size`.
   - multiview method면 `require_multiview: true`를 남긴다.
7. unit test를 추가한다.
   - registry resolve/build test.
   - descriptor required view test.
   - tensor-level step 계산 test.
   - stateful method면 lifecycle/configure test.
8. runner smoke를 추가할지 판단한다.
   - 기존 weak/strong loader만 쓰면 보통 필요 없다.
   - 새 view builder나 manifest field가 있으면 runner-level test를 추가한다.

## Algorithm skeleton

```python
from collections.abc import Mapping
from typing import Any

from torch import Tensor

from methods.ssl.base import QuerySslRequiredViews, QuerySslStepResult, TextBatchClassifier
from methods.ssl.registry import register_query_ssl_algorithm


class MyAlgorithm:
    algorithm_name = "my_algorithm"

    @property
    def uses_labeled_batches(self) -> bool:
        return True

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        if train_loader_length == 0:
            raise ValueError("MyAlgorithm labeled train_loader must not be empty.")
        if unlabeled_loader_length == 0:
            raise ValueError("MyAlgorithm unlabeled_loader must not be empty.")

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
    ) -> QuerySslStepResult:
        ...


@register_query_ssl_algorithm(
    "my_algorithm",
    display_name="MyAlgorithm",
    required_views=QuerySslRequiredViews(
        view_names=("weak_text",),
        view_builder_name="usb_weak",
    ),
)
def build_my_algorithm(parameters: Mapping[str, Any]) -> MyAlgorithm:
    return MyAlgorithm(...)
```

프로젝트 규칙상 실제 파일에는 `...`를 남기지 않는다. skeleton은 추가 위치와
interface shape만 보여주는 참고다.

## 어디를 건드리지 않을지

- `scripts/experiments/central_ssl_control/train_lora_<method>.py`를 새로 만들지 않는다.
  같은 runner로 표현되는 방법론은 Hydra `strategy_axes/ssl/consistency_method`만
  교체한다.
- `agent/`와 `main_server/`에 중앙 SSL method-specific 파일을 추가하지 않는다.
- `shared/`에는 payload shape가 실제로 바뀔 때만 손댄다.
- `methods/ssl/hooks/`에는 두 개 이상 algorithm에서 의미가 안정된 hook만 승격한다.
- 단일 method helper는 먼저 `methods/ssl/algorithms/<method>/` 안에 둔다.

## 비-USB 계열 판단 기준

USB/FixMatch 계열이 아니어도 아래 조건이면 쉽게 추가된다.

- 한 optimizer step에서 필요한 입력이 labeled batch와 unlabeled batch 안에 있다.
- model 호출은 algorithm이 `compute_step()` 안에서 직접 제어할 수 있다.
- 필요한 state가 algorithm instance 내부, `configure_training`, `configure_dataset`
  정도로 닫힌다.
- unlabeled row surface가 `weak` 또는 `weak/strong`으로 충분하다.

아래 조건이면 쉬운 추가가 아니다.

- tokenizer 전 단계에서 text를 합성하거나 두 문장/라벨을 mix해야 한다.
- optimizer step 전후로 teacher EMA update, adversarial perturbation, 별도 scheduler
  update가 필요하다.
- classification objective와 다른 pretraining phase가 필요하다.
- selection/evaluation cadence 자체를 바꿔야 한다.
- raw query retention, private state, FL client partition 같은 runtime 경계를 건드린다.

이 경우는 method 파일을 억지로 추가하기보다 먼저 필요한 새 seam을 명명한다. 예를 들어
`view_builder`, `phase_runner`, `post_optimizer_step`, `teacher_state_update`처럼
capability 이름으로 확장하고, method identity와 objective 의미는 여전히
`methods/ssl` 또는 `methods/federated_ssl`에 둔다.

## 검증 명령

작은 Query SSL method 추가 후 기본 검증은 아래처럼 닫는다.

```bash
uv run pytest tests/unit/test_methods_<method_name>.py tests/unit/test_scripts_hydra_configs.py
```

기존 weak/strong runner와 manifest까지 확인하려면 관련 runner test를 함께 실행한다.

```bash
uv run pytest tests/unit/test_lora_fixmatch_runner.py tests/unit/test_scripts_hydra_configs.py
```

구조 규칙을 확인한다.

```bash
uv run pytest tests/architecture/test_layer_dependencies.py
```
