# 2026-03-28 Phase 2 Study Guidance

## User

지금 밑에거에서 2단계 시작했는데 해당 주소의 코드까지 다 이해해야해? 아니면 넘어가? 구현까지 하나?

# 🚀 TraceMind 기준 필수 공부 리스트 (실전용)

---

## 🟢 Phase 1. Federated Analytics (출발점)

|주제|자료|왜 중요한가|링크|
|---|---|---|---|
|개념|Federated Analytics 소개|raw data 없이 통계만 학습하는 구조 이해|[https://research.google/blog/federated-analytics-collaborative-data-science-without-data-collection/](https://research.google/blog/federated-analytics-collaborative-data-science-without-data-collection/)|
|프레임워크|TensorFlow Federated Overview|client-server 구조 이해|[https://www.tensorflow.org/federated](https://www.tensorflow.org/federated)|
|튜토리얼|TFF Tutorials Overview|전체 흐름 한 번에 이해|[https://www.tensorflow.org/federated/tutorials/tutorials_overview](https://www.tensorflow.org/federated/tutorials/tutorials_overview)|

👉 핵심:

raw data ❌ → summary만 전송 ⭕

---

## 🔴 Phase 2. Aggregation (프로젝트 핵심)

| 주제  | 자료                    | 왜 중요한가                            | 링크                                                                                                                                           |
| --- | --------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 핵심  | Custom Aggregators    | WindowSummary → NormPack 설계 직접 연결 | [https://www.tensorflow.org/federated/tutorials/custom_aggregators](https://www.tensorflow.org/federated/tutorials/custom_aggregators)       |
| 사례  | Private Heavy Hitters | federated analytics 실제 패턴         | [https://www.tensorflow.org/federated/tutorials/private_heavy_hitters](https://www.tensorflow.org/federated/tutorials/private_heavy_hitters) |

👉 핵심:

local summary → global distribution

---

## 🔴 Phase 3. 통계 + 변화 감지 (진짜 중요, 지금 빠진 부분)

|주제|자료|왜 중요한가|링크|
|---|---|---|---|
|개념|Change Detection|너 프로젝트의 본질 (이상/변화 감지)|[https://en.wikipedia.org/wiki/Change_detection](https://en.wikipedia.org/wiki/Change_detection)|
|개념|Exponential Moving Average|baseline 계산 핵심|[https://en.wikipedia.org/wiki/Moving_average](https://en.wikipedia.org/wiki/Moving_average)|
|개념|Z-score|norm 대비 deviation 계산|[https://en.wikipedia.org/wiki/Standard_score](https://en.wikipedia.org/wiki/Standard_score)|
|개념|Outlier Detection|spike filtering|[https://en.wikipedia.org/wiki/Outlier](https://en.wikipedia.org/wiki/Outlier)|
|개념|Robust Statistics|중앙 aggregation 안정성|[https://en.wikipedia.org/wiki/Robust_statistics](https://en.wikipedia.org/wiki/Robust_statistics)|

👉 핵심:

이건 ML 문제가 아니라 "통계 + 신호처리 문제"

---

## 🟡 Phase 4. Local Decision Logic

|주제|자료|왜 중요한가|링크|
|---|---|---|---|
|개념|Time Series Analysis|baseline + trend 분석|[https://en.wikipedia.org/wiki/Time_series](https://en.wikipedia.org/wiki/Time_series)|
|개념|Trend Analysis|slope 계산|[https://en.wikipedia.org/wiki/Trend_analysis](https://en.wikipedia.org/wiki/Trend_analysis)|
|개념|Signal Processing|persistence / noise 제거|[https://en.wikipedia.org/wiki/Signal_processing](https://en.wikipedia.org/wiki/Signal_processing)|

👉 핵심:

Self-baseline + Norm + Persistence

---

## 🟡 Phase 5. FL 이론 (나중)

|주제|자료|왜 중요한가|링크|
|---|---|---|---|
|기본|FedAvg|FL 기본 구조|[https://proceedings.mlr.press/v54/mcmahan17a](https://proceedings.mlr.press/v54/mcmahan17a)|
|Survey|Advances and Open Problems in Federated Learning|전체 정리|[https://www.nowpublishers.com/article/Details/MAL-083](https://www.nowpublishers.com/article/Details/MAL-083)|
|최적화|Federated Optimization Guide|학습 안정성|[https://arxiv.org/abs/2107.06917](https://arxiv.org/abs/2107.06917)|

👉 핵심:

FL = optional (지금 당장 필요 없음)

---

## 🟡 Phase 6. 구조 확장 (중요하지만 나중)

|주제|자료|왜 중요한가|링크|
|---|---|---|---|
|구조|Federated Reconstruction|local vs global 분리|[https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization)|

---

## 🟡 Phase 7. 실습 프레임워크

|주제|자료|왜 중요한가|링크|
|---|---|---|---|
|연구용|TensorFlow Federated GitHub|구조 이해|[https://github.com/google-parfait/tensorflow-federated](https://github.com/google-parfait/tensorflow-federated)|
|입문|Flower|간단한 실습|[https://flower.ai/docs/framework/index.html](https://flower.ai/docs/framework/index.html)|
|실전|NVIDIA FLARE|production 환경|[https://nvflare.readthedocs.io/en/main/welcome.html](https://nvflare.readthedocs.io/en/main/welcome.html)|

---

## ⚫ Phase 8. Privacy (마지막)

|주제|자료|왜 중요한가|링크|
|---|---|---|---|
|개념|Secure Aggregation|privacy 핵심|[https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/](https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/)|
|개념|Differential Privacy|noise 기반 보호|[https://www.cis.upenn.edu/~aaroth/privacybook.html](https://www.cis.upenn.edu/~aaroth/privacybook.html)|

## Assistant

아니. 지금은 "해당 주소의 코드 전체"를 다 이해할 필요도 없고, TFF 내부 구현까지 파고들 필요도 없다. TraceMind 기준으로는 Phase 2의 목적이 TFF를 구현하는 게 아니라 `WindowSummary -> cohort aggregation -> NormPack` 경계를 정확히 이해하는 것이다.

지금 저장소 문서도 같은 방향이다. 현재 MVP 본체는 analytics 레일이고, `WindowSummary`를 중앙에서 집계해 `NormPack`을 만드는 단계가 핵심이다. 그리고 optional FL 레일은 그 위에 나중에 얹는 확장 계층으로 분리돼 있다. 따라서 공부 리스트의 "Phase 2"는 TFF 프레임워크 숙련이 아니라 집계 인터페이스와 통계 설계를 배우는 단계로 해석하는 게 맞다.

실전 기준으로는 이렇게 보면 된다.

1. `custom_aggregators`는 끝까지 읽되, 목표는 "AggregationProcess가 입력, 상태, 결과, 측정값을 어떻게 다루는지"를 이해하는 것까지다.
2. 튜토리얼의 모든 셀을 완벽히 설명하거나, TFF source code까지 따라갈 필요는 없다.
3. `private_heavy_hitters`는 사례 패턴만 가져오면 된다. IBLT 내부 알고리즘이나 문자열 heavy hitter 복원 세부 구현까지 들어갈 필요는 없다.
4. 구현은 "튜토리얼 재구현"이 아니라 현재 저장소의 중앙 집계 서비스에 맞는 단순한 aggregator를 만드는 쪽이 맞다.

즉 지금 필요한 최소 이해 수준은 아래 세 가지다.

1. 로컬이 무엇을 보내는가: `WindowSummary`
2. 중앙이 무엇을 계산하는가: cohort별 mean, sigma, median, mad, prevalence 같은 robust stats
3. 중앙이 무엇을 돌려주는가: `NormPack`

이 세 가지를 네 말로 설명할 수 있으면 넘어가도 된다. 반대로 TFF notebook의 각 코드 셀을 다 암기하거나, GitHub 내부 구현까지 읽는 건 지금 단계에서는 과하다.

구현 여부는 "예, 하지만 TFF 구현이 아니라 TraceMind 구현"이다. 현재 중앙 집계와 norm pack 서비스는 아직 자리표시자라서, 구현이 필요하다면 여기에 직접 `WindowSummary` 입력을 받아 cohort 통계를 계산하는 로직을 넣는 게 맞다. TFF custom aggregator를 바로 프로젝트 안에 들여오는 건 지금 단계에선 아니다.
