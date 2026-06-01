# Query SSL USB Extension Plan

## Continuation Prompt

다른 채팅에서 이어갈 때는 아래 prompt를 그대로 붙여 넣는다.

```text
TraceMind repo `/home/jmgjmg102/tracemind_server`에서 중앙 Query SSL에 USB 계열
method를 계속 가져올 수 있는 구조 리팩터링을 이어간다.

먼저 아래 파일을 읽어라.
1. AGENTS.md
2. docs/ai_context_manifest.yaml
3. docs/execution_index.md
4. QUERY_SSL_USB_EXTENSION_PLAN.md
5. methods/ssl/NEW_METHOD.md
6. methods/ssl/README.md
7. methods/adaptation/query_text_views/README.md
8. methods/adaptation/peft_text_encoder/training/loops.py
9. methods/adaptation/text_encoder_classifier/modeling.py

목표는 CoMatch 하나를 임시로 끼우는 것이 아니라, USB 계열 method를 계속 추가할 수
있도록 중앙 Query SSL runtime 경계를 깊게 만드는 것이다. CoMatch는 첫 acceptance
method로 구현한다. 새 method 추가 때문에 scripts runner가 method 이름으로 분기하면
구조 실패로 본다.

구현 순서:
1. Query SSL USB capability matrix 문서/테스트 기준 확정
2. QuerySslAlgorithmDescriptor에 batch/model/lifecycle capability metadata 추가
3. QuerySslStepContext 추가
4. usb_weak_strong_pair view surface 추가
5. projection head model extension lifecycle 추가
6. queue distribution alignment hook 추가
7. CoMatch core 구현
8. Hydra leaf, unit tests, architecture guard, docs sync

주의:
- USB를 무조건 1:1 복사하지 않는다. 핵심 수식과 state lifecycle은 보존하고,
  TraceMind의 method/conf/runtime 경계에 맞게 포팅한다.
- `agent/`, `main_server/`, `shared/`는 중앙 USB SSL method 추가만으로 건드리지 않는다.
- projection head는 optimizer/checkpoint lifecycle에 반드시 포함한다.
- memory/DA/threshold state는 algorithm state로 저장한다.
- CoMatch 기준 USB source는 `microsoft/Semi-supervised-learning`
  commit `1ef4cbebcc0b368158315aeb425053858cf6c845`의
  `semilearn/algorithms/comatch/comatch.py`다.
```

## 판단

기존 CoMatch plan은 `methods/ssl/algorithms/comatch/`에 파일을 추가하는 방향은 맞지만,
현재 중앙 Query SSL Interface가 CoMatch와 이후 USB method 전체를 받기에는 좁다.

현재 구조는 FixMatch, FlexMatch, FreeMatch, AdaMatch처럼 logits 기반 weak/strong
consistency 계열에는 충분히 열려 있다. 반면 CoMatch, SimMatch, CRMatch처럼 feature
projection, memory queue, dual strong view, model lifecycle 변경이 필요한 method에는
작은 우회가 쌓일 위험이 있다.

따라서 우선순위는 다음과 같다.

```text
Query SSL runtime capability 확장
-> CoMatch를 첫 acceptance method로 구현
-> 이후 USB method를 capability 조합으로 추가
```

## 목표

새 USB method를 추가할 때 기본 수정 범위는 아래로 끝나야 한다.

```text
methods/ssl/algorithms/<method>/
conf/strategy_axes/ssl_objective/consistency_method/<method>_usb_v1.yaml
tests/unit/test_methods_<method>.py
```

단, 새 method가 아직 없는 capability를 요구하면 해당 capability owner만 한 번 연다.
이후 같은 capability를 쓰는 method는 runner를 다시 수정하지 않는다.

성공 기준:

- `scripts/support/query_ssl_peft/runners/consistency.py`가 method 이름으로 분기하지 않는다.
- descriptor가 required batch/model/lifecycle capability를 표현한다.
- projection head 같은 trainable extension은 optimizer와 checkpoint에 포함된다.
- memory queue, distribution alignment queue, adaptive threshold state는 algorithm state로 저장된다.
- 중앙 SSL method 추가만으로 `agent/`, `main_server/`, `shared/`를 건드리지 않는다.
- Hydra leaf는 조합과 파라미터만 소유하고, 수식 의미는 `methods/ssl`이 소유한다.

## USB Method Capability 분류

USB method를 모두 같은 `compute_step(model, labeled_batch, unlabeled_batch)`로 보면
구조가 막힌다. method를 capability 기준으로 나눈다.

| 등급 | 예 | 필요한 경계 |
|---|---|---|
| A. logits-only weak/strong | FixMatch, FreeMatch 일부 | 현재 구조로 가능 |
| B. stateful threshold/DA | FlexMatch, FreeMatch, AdaMatch, SoftMatch | algorithm state export/load, DA hook |
| C. feature/projection 필요 | CoMatch, SimMatch, CRMatch | feature extraction, projection head lifecycle |
| D. dual/multi strong view 필요 | CoMatch, ReMixMatch 계열 | batch view surface 확장 |
| E. optimizer/model lifecycle 변경 | MeanTeacher, VAT, MixMatch/ReMixMatch 일부 | teacher/EMA, post-step hook, adversarial step, mix batch support |

현재 구현은 A/B에 가깝다. CoMatch를 제대로 받으려면 C/D를 먼저 연다. E는 CoMatch 이후
별도 단계로 연다.

## 리팩터링 계획

### 1. Descriptor를 capability 중심으로 확장

`methods/ssl/base.py`의 `QuerySslAlgorithmDescriptor`가 view만 표현하는 상태에서,
method가 요구하는 runtime surface를 함께 표현하게 한다.

초기 후보:

```python
required_batch_surface: "weak" | "weak_strong" | "weak_strong_pair"
required_model_outputs: frozenset[str]  # 예: {"logits", "pooled_features"}
model_extension: str | None             # 예: "projection_head"
lifecycle_requirements: frozenset[str]  # 예: {"dataset_state", "step_context"}
```

이 값은 public Hydra axis가 아니라 method descriptor metadata다. runner는 method 이름을
보지 않고 capability만 본다.

### 2. Step context 추가

USB method는 `epoch`, `it`, `num_train_iter`, `num_classes`, `device` 같은 runtime
값을 자주 쓴다. `configure_training`과 `configure_dataset`만으로는 queue warm-up,
ramp-up, step-dependent policy가 흐릿해진다.

초기 후보:

```python
@dataclass(frozen=True, slots=True)
class QuerySslStepContext:
    epoch_index: int
    step_index: int
    global_step: int
    total_train_steps: int
    num_classes: int
    device: torch.device
```

기존 algorithm은 context를 무시할 수 있게 compatibility shim을 둔다. 단기적으로는
`compute_step` 호출에서 optional keyword를 duck-typing으로 넘기고, 다음 단계에서
Protocol을 정리한다.

### 3. View surface 확장

현재 `usb_multiview`는 `aug_0`과 `aug_1` 중 하나만 골라 `strong_input_ids`로 만든다.
CoMatch는 두 strong view가 모두 필요하다.

새 surface:

```text
usb_weak
usb_weak_strong
usb_weak_strong_pair
```

`usb_weak_strong_pair` batch:

```text
weak_input_ids
weak_attention_mask
strong_0_input_ids
strong_0_attention_mask
strong_1_input_ids
strong_1_attention_mask
```

Owner는 `methods/adaptation/query_text_views/`다. `scripts`는 descriptor의
`view_builder_name` 또는 `required_batch_surface`를 넘기기만 한다.

### 4. Model extension lifecycle 추가

CoMatch projection head를 algorithm-local `nn.Module`로 들고 있으면 현재 optimizer가
학습하지 못한다. projection head는 model extension으로 붙여 optimizer와 checkpoint에
포함해야 한다.

초기 위치:

```text
methods/ssl/model_capabilities.py
methods/adaptation/peft_text_encoder/training/ssl_model_extensions.py
```

흐름:

```text
algorithm descriptor resolve
-> required model extension 적용
-> optimizer 생성
-> training loop 실행
-> checkpoint가 model + optimizer + algorithm_state 저장
```

feature extraction은 현재 `TextEncoderWithLinearHead.extract_pooled_features(...)`가
있으므로, CoMatch는 method-local `FeatureExtractingClassifier` Protocol로 요구한다.
두 개 이상 method가 같은 요구를 안정적으로 공유하면 `methods/ssl/model_capabilities.py`
로 승격한다.

### 5. Distribution alignment hook 정리

현재 AdaMatch용 EMA DA는 있지만, USB CoMatch는 queue DA를 쓴다. 같은 파일 안에서
역할별 구현을 분리한다.

```text
methods/ssl/hooks/distribution_alignment.py
  DistAlignEmaHook
  DistAlignQueueHook
```

DA 선택은 algorithm이 소유한다. 별도 user-facing `teacher_provider`나 DA strategy axis를
먼저 열지 않는다.

### 6. USB provenance 표준화

USB를 무조건 1:1 복사하지 않는다. 대신 각 method README 또는 `original_spec.py`에
원본 기준과 의도적 차이를 남긴다.

필수 항목:

```text
source_repo
source_commit
source_path
preserved_core
intentional_deviations
unsupported_usb_features
```

CoMatch 기준:

```text
source_repo: microsoft/Semi-supervised-learning
source_commit: 1ef4cbebcc0b368158315aeb425053858cf6c845
source_path: semilearn/algorithms/comatch/comatch.py
```

## CoMatch v1 범위

CoMatch는 새 구조의 첫 acceptance method다.

보존할 핵심:

- supervised CE
- weak pseudo-label probability
- queue distribution alignment
- flat memory queue 기반 probability smoothing
- confidence mask
- strong_0 consistency loss
- strong_0/strong_1 graph contrastive loss
- projection head와 L2-normalized embedding

v1에서 제외 가능한 항목:

- distributed `concat_all_gather`
- EPASS multi-projection
- USB `AlgorithmBase` wrapper 구조
- EMA model pseudo-label source

예상 파일:

```text
methods/ssl/algorithms/comatch/
  README.md
  comatch.py
  memory_bank.py
  original_spec.py

conf/strategy_axes/ssl_objective/consistency_method/comatch_usb_v1.yaml
tests/unit/test_methods_comatch.py
```

CoMatch config는 USB parity와 ablation을 구분한다. USB parity preset에는
`temperature`, `p_cutoff`, `contrast_p_cutoff`, `queue_batch`, `smoothing_alpha`,
`da_len`, `proj_size`, `lambda_u`, `lambda_c`, `supervised_loss_weight`,
`unlabeled_batch_size`, `require_multiview`를 둔다.

## 구현 순서

1. Query SSL USB capability matrix 문서 또는 README 섹션 추가
2. descriptor capability metadata 추가
3. 기존 FixMatch/FlexMatch/FreeMatch/AdaMatch descriptor test 갱신
4. `QuerySslStepContext` 추가
5. `usb_weak_strong_pair` dataset/dataloader surface 추가
6. projection head model extension lifecycle 추가
7. queue distribution alignment hook 추가
8. CoMatch memory bank와 tensor-level loss 구현
9. CoMatch algorithm adapter 구현
10. Hydra leaf와 compose test 추가
11. training loop checkpoint/optimizer 검증 추가
12. architecture guard로 scripts method-name 분기 금지 고정

## 검증 계획

우선 검증:

```bash
uv run pytest tests/unit/test_methods_comatch.py
uv run pytest tests/unit/test_query_text_views_data.py
uv run pytest tests/unit/test_peft_encoder_training_core.py
uv run pytest tests/unit/test_scripts_hydra_configs.py
uv run pytest tests/architecture/test_layer_dependencies.py
```

smoke는 구조가 닫힌 뒤 reduced budget으로 실행한다.

```bash
uv run pytest tests/unit/test_peft_fixmatch_runner.py
```

CoMatch full run은 method/core wiring과 metadata가 smoke로 확인된 뒤 별도 실행한다.

## Open Decisions

1. `QuerySslStepContext`를 바로 Protocol에 넣을지, compatibility duck-typing으로
   한 단계 열지 결정한다.
2. projection head를 generic model extension으로 둘지, CoMatch wrapper로 먼저 시작할지
   결정한다. 권장은 generic model extension이다.
3. `usb_multiview` 이름을 유지할지, `usb_weak_strong`으로 새 이름을 열고 기존 이름을
   compatibility alias로 둘지 결정한다.
4. MeanTeacher/VAT/MixMatch 계열을 언제 E등급 lifecycle 리팩터링으로 열지 결정한다.
