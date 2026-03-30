# 2026-03-31 FL result interpretation

## User

```text
initial_model_revision=sim_rev_0000
initial_prototype_version=proto_sim_0000
initial_validation=accuracy:0.7404,accepted_ratio:0.5392
final_model_revision=sim_rev_0003
final_prototype_version=proto_sim_0003
final_validation=accuracy:0.7402,accepted_ratio:0.5410
round_count=3

fl 돌린결과인데 어떻게 해석해?
```

## Assistant

```text
현재 활성 문서 기준으로 이 결과는 shared representation 또는 shared adapter를
연합학습으로 조금 갱신한 뒤 validation이 어떻게 변했는지 보는 수치로 해석하는 게 맞다.

핵심 해석은 아래와 같다.

1. revision 진행
- model revision: sim_rev_0000 -> sim_rev_0003
- prototype version: proto_sim_0000 -> proto_sim_0003
- round_count=3 이므로 라운드가 돌 때마다 새 shared state와 prototype artifact가
  발행된 것으로 읽을 수 있다.

2. 성능 변화
- accuracy: 0.7404 -> 0.7402
- 절대 변화량: -0.0002
- 사실상 거의 동일한 수준이다.

- accepted_ratio: 0.5392 -> 0.5410
- 절대 변화량: +0.0018
- acceptance가 아주 미세하게 올라갔다.

3. 의미
- 3라운드 동안 모델이 크게 좋아졌다고 보기는 어렵다.
- 그렇다고 성능이 무너진 것도 아니다.
- 즉 지금 FL update는 shared layer를 약간 바꿨지만 validation 기준으로는
  거의 보합이다.

4. 조금 더 실무적으로 말하면
- accuracy는 그대로인데 accepted_ratio만 조금 올랐으므로,
  모델이 약간 더 permissive해졌을 가능성이 있다.
- 다만 변화폭이 너무 작아서 validation set 크기, seed variance,
  client sampling variance를 모르면 유의미한 차이라고 단정하면 안 된다.

5. 지금 단계에서의 좋은 신호
- revision이 3번 올라갔는데 accuracy drift가 거의 없다.
- 이는 aggregation이 불안정하게 폭주하진 않았다는 뜻으로 볼 수 있다.

6. 지금 단계에서의 아쉬운 신호
- "FL을 돌렸더니 성능이 개선됐다"라고 말할 근거는 없다.
- 현재 local objective, aggregation weight, learning rate, client heterogeneity 처리 중
  하나 이상이 아직 개선폭을 만들 만큼 강하지 않을 수 있다.

7. 다음으로 꼭 봐야 할 것
- round별 accuracy/accepted_ratio 추이
- client별 before/after 편차
- confusion matrix 또는 precision/recall 변화
- seed를 바꿔도 같은 경향이 나오는지

한 줄 요약:
이번 결과는 "3라운드 FL 후 shared model/prototype revision은 갱신됐지만,
validation 성능은 사실상 보합이고 acceptance만 아주 미세하게 증가했다"로
해석하는 게 가장 안전하다.
```

## User

```text
진행해봐
```

## Assistant

```text
같은 실험의 round별 흔적과 metric 산출 경로를 먼저 찾겠다고 안내했다.
`run_federated_simulation.py`, 설정 YAML, `tmp/federated_simulation` 산출물,
validation 크기, 누적된 update 파일 패턴을 확인했다.

이후 4,961건 validation 전체를 재평가하려고 `.venv/bin/python`으로
재계산을 시작했는데, 로컬 런타임이 `cpu`로 잡혀 CPU 배치 인퍼런스가 진행되었다.
```

## User

```text
난 cpu로 안 돌릴거야 cpu 실행중인거 종료시켜
```

## Assistant

```text
내가 띄운 `.venv/bin/python` PTY 세션에 `Ctrl-C`를 보내 CPU 실행을 중단했다.
이후 `ps -p 70213,70214 -o pid=,stat=,comm=`로 확인했을 때 남아 있는 프로세스가
없어서 종료된 상태임을 확인했다.
```

## User

```text
내가 실행했었던거 알려줄게 이걸 토대로 tmp에서 기록 찾아서 들어가서 확인해봐

(tracemind-server) jmgjmg102@DESKTOP-VVOL8MH:~/tracemind_server$   uv run python scripts/experiments/run_federated_simulation.py \
    federated_run_preset=standard \
    federated_run_preset.rounds=3 \
    federated_run_preset.client_count=4
...
initial_model_revision=sim_rev_0000
initial_prototype_version=proto_sim_0000
initial_validation=accuracy:0.7404,accepted_ratio:0.5392
final_model_revision=sim_rev_0003
final_prototype_version=proto_sim_0003
final_validation=accuracy:0.7402,accepted_ratio:0.5410
round_count=3
```

## Assistant

```text
재실행 없이 `tmp/federated_simulation` 산출물만 읽어서 2026-03-31 02:30~02:35 KST에
해당하는 파일 구간을 분리했다. 같은 디렉터리에 02:17, 02:26 실행 흔적도 함께
남아 있어서, 이번 run은 02:30 이후 파일만 기준으로 봐야 한다.

확인한 핵심은 아래와 같다.

1. 이번 run의 main_server 산출물 타임라인
- 02:30:15 `sim_rev_0000` state 저장
- 02:30:29 `sim_rev_0000` manifest와 `proto_sim_0000` 저장
- 02:31:52 `sim_rev_0001` / `proto_sim_0001`
- 02:33:27 `sim_rev_0002` / `proto_sim_0002`
- 02:35:04 `sim_rev_0003`
- 02:35:18 `proto_sim_0003`

2. batch 로그와 실제 shard 크기가 정확히 대응한다
- bootstrap rows: 8,930 -> 559 batches
- validation rows: 4,961 -> 311 batches
- agent_01 shard: 4,990 -> 312 batches
- agent_02 shard: 8,824 -> 552 batches
- agent_03 shard: 12,542 -> 784 batches
- agent_04 shard: 9,365 -> 586 batches

3. 모든 라운드에서 4개 client가 모두 update를 냈고, 각 client는 매번 64개씩만 반영됐다
- 이는 `standard` preset의 `max_examples=64` cap과 맞는다.
- round당 총 반영 예시는 256개다.

4. 더 중요한 점은 update 조성이 3라운드 내내 사실상 고정돼 있었다
- round_0001 totals: anxiety 14, depression 71, suicidal 171
- round_0002 totals: anxiety 14, depression 71, suicidal 171
- round_0003 totals: anxiety 14, depression 71, suicidal 171
- `normal` pseudo-label은 3라운드 전체에서 한 번도 선택되지 않았다.

5. shard 분포와 비교하면 selection 편향이 보인다
- agent_03 shard는 depression 2,611 / normal 9,931인데 update는 depression 37 / suicidal 23 / anxiety 4
- agent_04 shard는 normal 3,311 / suicidal 6,054인데 update는 매 라운드 suicidal 64 only
- 즉 현재 threshold + heuristic update 조합은 normal을 거의 학습 신호로 쓰지 못하고 있다.

6. shared adapter drift는 누적되지만 작다
- sim_rev_0001: mean_abs_from_rev0 0.001061, max_abs 0.009688
- sim_rev_0002: mean_abs_from_rev0 0.002122, max_abs 0.019467
- sim_rev_0003: mean_abs_from_rev0 0.003183, max_abs 0.029337
- 1024차원 중 below 1.0 = 518, above 1.0 = 506 으로 부호 패턴도 사실상 유지된다.

7. prototype도 매 라운드 재생성되지만 이동량은 매우 작다
- sample_count는 전 버전 동일: anxiety 991, depression 2611, normal 3310, suicidal 2018
- `proto_sim_0000 -> proto_sim_0003` centroid cosine
  - anxiety 0.99995150
  - depression 0.99995184
  - normal 0.99994315
  - suicidal 0.99995349
- 즉 prototype version은 올라가지만 위치 변화는 미세하다.

8. 따라서 이번 flat metric은 tmp 기록만 봐도 설명된다
- 각 round의 client update가 거의 같은 분포와 비슷한 confidence/margin으로 반복된다.
- adapter와 prototype drift도 작다.
- 그래서 3라운드가 서로 다른 학습 단계라기보다, 거의 같은 update를 세 번 누적 적용한 것에 가깝다.
```

## User

```text
normal을 학습신호로 왜 못 쓰는데?
```

## Assistant

```text
코드상으로 `normal`을 금지하는 로직은 없다. 문제는 현재 pseudo-label selection 정책이
`normal`을 거의 최종 채택에 못 올린다는 데 있다.

근거:

1. [`agent/src/services/training/pseudo_label_service.py`](/home/jmgjmg102/tracemind_server/agent/src/services/training/pseudo_label_service.py)
   - top1 label만 pseudo-label 후보가 된다.
   - `confidence >= 0.6` and `margin >= 0.02`를 만족해야 accepted 된다.
   - 그 뒤 accepted 후보를 confidence, margin 순으로 정렬해서 `max_examples`까지만 남긴다.

2. [`scripts/conf/experiments/run_federated_simulation.yaml`](/home/jmgjmg102/tracemind_server/scripts/conf/experiments/run_federated_simulation.yaml)
   - 이번 run threshold는 `confidence_threshold=0.6`, `margin_threshold=0.02`

3. [`scripts/conf/federated_run_preset/standard.yaml`](/home/jmgjmg102/tracemind_server/scripts/conf/federated_run_preset/standard.yaml)
   - client당 최대 반영 예시는 `64`

4. 따라서 `normal`이 빠질 수 있는 지점은 두 군데다.
   - 아예 threshold를 못 넘는다.
   - threshold는 넘겨도 confidence/margin 순위에서 top 64 밖으로 밀린다.

5. 이번 run 산출물은 실제로 두 번째 현상까지 강하게 시사한다.
   - agent_03 shard는 `normal 9,931`이었는데 round_0001 update는 `depression 37`, `suicidal 23`, `anxiety 4`, `normal 0`
   - agent_04 shard는 `normal 3,311 / suicidal 6,054`인데 round_0001 update는 `suicidal 64` only
   - 즉 `normal`이 많아도, 현재 점수 체계에서 더 높은 confidence/margin을 받은 건 mostly depression/suicidal이었다.

6. 중요한 보정:
   - 지금 기록만으로는 `normal`이 threshold 자체를 거의 못 넘었는지,
     아니면 넘긴 뒤에도 top 64에서 탈락했는지까지는 확정할 수 없다.
   - 그걸 알려면 per-row score/confidence/margin dump가 필요하다.

따라서 정확한 표현은
`normal을 못 쓴다`가 아니라
`현재 prototype scoring + threshold + top64 cap 조합에서 normal이 거의 채택되지 않는다`가 맞다.
```

## User

```text
다시 돌렸는데 이제 왜 노멀이 제대로 잡히지 않았었는지 다시 점검해봐
```

## Assistant

```text
최신 run은 2026-03-31 03:08~03:13 KST 구간 파일로 확인했다.
이번에는 `selection_dumps`가 있어서 원인을 더 직접적으로 확인할 수 있었다.

핵심 결론:

1. `normal`의 주된 문제는 threshold rejection이 아니라 cap 탈락이다.
- round_0001 전체 기준 true normal:
  - accepted: 2
  - dropped_by_cap: 8,669
  - threshold_rejected: 4,571
  - total: 13,242
- round_0003도 거의 동일:
  - accepted: 2
  - dropped_by_cap: 8,720
  - threshold_rejected: 4,520

즉 true normal의 약 65% 이상은 threshold는 넘겼지만 top-64 안에 못 들어가서 잘렸다.

2. predicted normal도 거의 전부 cap에서 잘린다.
- round_0001 predicted normal:
  - dropped_by_cap: 8,931
  - threshold_rejected: 3,802
  - accepted: 0
- round_0003 predicted normal:
  - dropped_by_cap: 8,985
  - threshold_rejected: 3,750
  - accepted: 0

즉 normal로 예측된 샘플조차 최종 accepted로 한 건도 못 들어갔다.

3. agent별로 보면 더 선명하다.
- agent_03 true normal 9,931 중
  - accepted 2
  - dropped_by_cap 6,523
  - threshold_rejected 3,406
- agent_04 true normal 3,311 중
  - accepted 0
  - dropped_by_cap 2,146
  - threshold_rejected 1,165

4. rank를 보면 `normal`은 top64에 근처도 못 간 경우가 많다.
- agent_03 round_0001 predicted normal dropped rank
  - min 355, median 4,389
- agent_04 round_0001 predicted normal dropped rank
  - min 1,431, median 3,537

즉 normal 후보가 threshold를 통과해도 confidence/margin 정렬에서 이미 훨씬 뒤에 있다.

5. accepted된 true normal 2개도 normal로 맞게 잡힌 게 아니다.
- agent_03 round_0001 / round_0003에서만 2개 있었고
  - 하나는 predicted anxiety, pre_cap_rank 54
  - 하나는 predicted depression, pre_cap_rank 23

즉 현재 top64에 들어오는 건 'normal을 정상적으로 포착한 예시'가 아니라,
normal인데 다른 클래스로 강하게 오예측된 예시다.

결론:
이제는 더 정확히 말할 수 있다.

`normal이 안 잡힌 주원인은 threshold보다 cap이다.`
현재 scoring에서는 normal 관련 후보가 threshold를 넘는 경우도 많지만,
confidence/margin 순위가 낮아서 top64에 거의 못 들어간다.
그나마 들어간 소수도 normal 정답 포착이 아니라 high-confidence 오예측이다.
```

## User

```text
왜 노멀은 상위에 못 가는데?
```

## Assistant

```text
최신 selection dump를 보면 이유가 더 분명하다.

1. 정렬 규칙상 confidence가 먼저고 margin은 그다음이다.
- `PseudoLabelSelectionService`는 threshold를 통과한 후보를
  `(-confidence, -margin, query_id)` 순으로 정렬한 뒤 top 64만 남긴다.
- 그래서 margin이 좋아도 confidence가 낮으면 상위에 못 간다.

2. normal의 핵심 문제는 margin이 아니라 confidence다.
- round_0001 전체 threshold-accepted 후보 기준:
  - predicted normal
    - count: 8,931
    - max_conf: 0.800512
    - median_conf: 0.670351
    - median_margin: 0.084775
    - best_rank: 355
  - predicted suicidal
    - count: 3,697
    - max_conf: 0.898272
    - median_conf: 0.799735
    - median_margin: 0.030297
    - best_rank: 1
  - predicted depression
    - count: 3,171
    - max_conf: 0.881519
    - median_conf: 0.781571
    - median_margin: 0.027599
    - best_rank: 3

즉 normal은 margin 자체는 나쁘지 않다. 오히려 depression/suicidal보다 median margin이 더 크다.
그런데 absolute top1 confidence가 훨씬 낮다. 정렬이 confidence 우선이라 여기서 밀린다.

3. 실제 cutoff와 비교하면 더 명확하다.
- agent_03 round_0001 rank64 cutoff:
  - predicted suicidal
  - confidence 0.841600
  - margin 0.027931
- 같은 agent에서 predicted normal 최고 confidence:
  - 0.800512
  - pre_cap_rank 355

- agent_04 round_0001 rank64 cutoff:
  - predicted suicidal
  - confidence 0.867483
  - margin 0.034095
- 같은 agent에서 predicted normal 최고 confidence:
  - 0.786043
  - pre_cap_rank 1431

즉 normal은 threshold는 넘겨도 top64 cutoff confidence 자체에 못 닿는다.

4. 왜 confidence가 낮은지는 현재 구조상 이렇게 해석하는 게 맞다.
- 이번 run 기본 prototype builder는 `single`이다.
- 즉 normal 전체를 하나의 centroid로 대표한다.
- normal 카테고리가 더 넓고 이질적이면 centroid가 평균화되어
  많은 normal 샘플이 "꽤 normal 같음" 정도의 중간 score는 받지만,
  depression/suicidal처럼 아주 높은 top1 cosine은 잘 못 만든다.

이 마지막 문장은 score 분포를 바탕으로 한 해석이다.
코드와 산출물이 직접 보여주는 사실은
`normal은 margin보다 confidence에서 밀리고, top64 cutoff confidence에 못 도달한다`
는 점이다.
```

## User

```text
지금 구조는 안 좋은거잖아 어떻게 개선 가능한데?
그리고 confidence는 뭔데?
쓰레시홀드 문제는 아니고 상위권 선정해서 라벨링 하는 과정에서 상위권 선정이 문제인거지?
```

## Assistant

```text
정리하면 아래와 같다.

1. confidence의 의미
- 현재 confidence는 확률이 아니다.
- `ScoringService`가 계산한 category별 cosine similarity 중 top1 score가 그대로 confidence다.
- 즉 "이 샘플이 정답일 확률 0.84"가 아니라
  "현재 임베딩이 winning prototype과 cosine 0.84"에 가깝다.
- label마다 score 분포가 다를 수 있는데, 지금은 그 raw score를 그대로 cross-label ranking에 쓴다.

2. threshold vs cap
- 최신 run 기준 true normal 13,242건 중
  - accepted 2
  - dropped_by_cap 8,669
  - threshold_rejected 4,571
- 따라서 primary bottleneck은 cap/top64 ranking이 맞다.
- 다만 threshold도 무시할 수준은 아니고, secondary bottleneck이다.

3. 왜 지금 구조가 안 좋냐
- raw cosine score는 label별로 calibration이 안 돼 있다.
- 그런데 selection은 label 구분 없이 global top64를 뽑는다.
- normal은 threshold를 넘겨도 absolute confidence가 상대적으로 낮아서 잘린다.
- single prototype 구조까지 겹쳐 normal 내부 분산이 centroid에 평균화되면
  high-confidence normal이 더 적어진다.

4. 개선 우선순위
- 1순위: selection policy를 global top64에서 label-aware policy로 바꾼다.
  - 예: predicted_label별 quota
  - 예: top16 per predicted label + leftover global fill
  - 예: label별 max/min cap

- 2순위: raw confidence 대신 calibrated ranking score를 쓴다.
  - 예: predicted_label 내부 percentile rank
  - 예: label별 z-score normalization
  - 예: validation 기반 label-specific confidence threshold

- 3순위: normal representation을 더 세분화한다.
  - 예: `prototype_builder=kmeans`
  - 예: normal만 multi-prototype 허용
  - 예: single centroid는 유지하되 selection에는 nearest prototype distance를 함께 사용

- 4순위: selection stage를 2단으로 나눈다.
  - stage A: threshold 통과
  - stage B: balance-aware sampling
  - 지금은 stage B가 raw confidence one-rule이라 bias가 너무 크다.

5. 지금 코드베이스에서 가장 자연스러운 구현 방향
- `TrainingSelectionPolicy`는 현재 `max_examples`와 `extras`만 있다.
- 여기에
  - `selection_mode`
  - `max_examples_per_label`
  - `min_examples_per_label`
  - `global_fill_examples`
  같은 필드를 `extras` 또는 정식 필드로 추가할 수 있다.
- 그리고 `PseudoLabelSelectionService.select()`에서
  threshold 통과 후보를 predicted_label별로 나눈 뒤 quota 기반으로 뽑으면 된다.

6. 그래서 네 질문에 대한 짧은 답
- yes, 지금 문제의 중심은 threshold보다 상위권 선정 로직이다.
- 더 정확히는
  `raw cosine confidence를 label 구분 없이 global ranking에 쓰는 구조`
  가 문제다.
```
