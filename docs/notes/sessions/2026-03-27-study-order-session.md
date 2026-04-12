# 2026-03-27 Study Order Session

## User

지금 내 목표가 있잖아 최종 목표까지 보았을 떄 여기있는거 공부하면 되나? 아니면 다른것도 공부해야 하나?
순서는?

1. 📘 입문 (Core Foundations)
구분	제목	설명	링크
논문	McMahan et al., 2017 (FedAvg)	FL의 출발점, 기본 알고리즘	https://proceedings.mlr.press/v54/mcmahan17a

Google Research	Federated Analytics 소개	데이터 수집 없이 분석하는 개념	https://research.google/blog/federated-analytics-collaborative-data-science-without-data-collection/

Docs	TensorFlow Federated Overview	FL + Analytics 전체 구조 소개	https://www.tensorflow.org/federated
2. 🧠 전체 그림 (Survey / Theory)
구분	제목	설명	링크
Survey	Kairouz et al., 2021	FL 표준 survey (거의 필독)	https://www.nowpublishers.com/article/Details/MAL-083

DOI	Kairouz et al. DOI	공식 DOI 링크	https://doi.org/10.1561/2200000083

arXiv	A Field Guide to Federated Optimization	최적화 관점에서 설명	https://arxiv.org/abs/2107.06917

arXiv	Konečný et al.	통신 효율 최적화 고전	https://arxiv.org/abs/1610.05492
3. 🎯 네 프로젝트와 직접 연결되는 자료 (핵심)
구분	제목	설명	링크
Tutorial	TFF Tutorials Overview	전체 흐름 한 번에 이해	https://www.tensorflow.org/federated/tutorials/tutorials_overview

Tutorial	Private Heavy Hitters	Federated Analytics 실제 사례	https://www.tensorflow.org/federated/tutorials/private_heavy_hitters

Tutorial	Custom Aggregators	중앙 aggregation 설계 핵심	https://www.tensorflow.org/federated/tutorials/custom_aggregators

Tutorial	Federated Reconstruction	일부 파라미터 로컬 유지 구조	https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization

👉 특히:

WindowSummary → NormPack = Custom Aggregation + Heavy Hitters 구조랑 거의 동일
4. 🛠️ 실습 프레임워크
구분	이름	설명	링크
Docs	TensorFlow Federated	공식 문서	https://www.tensorflow.org/federated

GitHub	TensorFlow Federated	FL 연구용 프레임워크	https://github.com/google-parfait/tensorflow-federated

Docs	Flower	입문/실습 최적	https://flower.ai/docs/framework/index.html

GitHub	Flower	간단한 FL 실험용	https://github.com/adap/flower

Docs	NVIDIA FLARE	production/cross-silo 환경	https://nvflare.readthedocs.io/en/main/welcome.html

GitHub	NVIDIA FLARE	엔터프라이즈 FL	https://github.com/NVIDIA/NVFlare

GitHub	FedML	다양한 예제 제공	https://github.com/FedML-AI/FedML

## Assistant

지금 목록은 좋은 시작이지만, TraceMind 최종 목표까지 보려면 그대로만 공부하는 것은 부족하다고 정리했다.

이유는 이 프로젝트의 기본 경로가 일반적인 FedAvg 중심 FL이 아니라, 먼저 로컬에서 `WindowSummary`를 만들고 중앙에서 `NormPack`을 만드는 analytics-first 구조이기 때문이다. 문서 기준으로는 `WindowSummary`, `NormPack`은 cohort parameter learning 산출물이고, decision-model parameter는 나중 FL 산출물로 분리되어 있다. 따라서 학습 순서도 FL 알고리즘 전체를 먼저 파는 방식보다, analytics 레일을 먼저 닫고 그 위에 optional FL 레일을 얹는 방식으로 가는 것이 맞다고 설명했다.

추천 순서는 아래처럼 정리했다.

1. Federated Analytics 관점 먼저 잡기
   - Federated Analytics 소개
   - TFF Overview
   - TFF Tutorials Overview
   - 이 단계에서는 "원문을 중앙으로 보내지 않고 어떤 요약만 보낼 것인가"를 이해하는 것이 핵심이다.

2. TraceMind 핵심 계약과 통계 흐름 이해
   - `WindowSummary`
   - `NormPack`
   - `AssessmentResult`
   - cohort grouping, mean/variance/prevalence, robust aggregation
   - 이 부분은 단순 FL보다 이 프로젝트에 더 직접적이다.

3. 로컬 판단에 필요한 시계열 기초 추가
   - self-baseline
   - slope / persistence
   - spike filtering
   - change point detection 또는 anomaly detection 기초
   - 이유는 TraceMind의 현재 MVP가 "분류기 학습"보다 "지속 변화 감지"에 더 가깝기 때문이다.

4. 중앙 aggregation 구조 학습
   - TFF Custom Aggregators
   - TFF Private Heavy Hitters
   - heavy hitters는 정확히 동일 구조는 아니지만, privacy-preserving analytics와 aggregation 사고방식을 익히는 데 매우 가깝다고 설명했다.

5. 그 다음에 FL 이론 본격 진입
   - FedAvg
   - Kairouz survey
   - Field Guide to Federated Optimization
   - Konečný communication-efficient 고전
   - 다만 이 단계는 "지금 당장 구현할 것"보다는 "나중 optional FL 레일을 열기 위한 이론 축"이라고 구분했다.

6. 부분 로컬 파라미터 개념 확인
   - Federated Reconstruction
   - 로컬에 남겨야 하는 것과 중앙이 모을 수 있는 것을 분리하는 사고에 도움된다고 설명했다.

7. privacy hardening은 마지막에 집중
   - secure aggregation
   - differential privacy
   - 필요 시 homomorphic encryption
   - 이 프로젝트 문서도 MVP 이전에 DP/HE를 넣지 말고, secure aggregation과 제한적 DP를 나중에 붙이는 순서를 권장한다고 설명했다.

추가로 별도 공부가 필요한 영역도 정리했다.

1. secure aggregation
2. differential privacy 기초
3. robust statistics / robust aggregation
4. time-series change detection
5. feedback 또는 self-report 신호 설계

프레임워크 우선순위도 나눠 설명했다.

1. 지금은 TFF가 가장 직접적이다.
   - 이유는 federated analytics, custom aggregation, heavy hitters, DP tutorial까지 한 흐름으로 보기 쉽기 때문이다.

2. Flower는 나중에 FL-ready runtime이나 orchestration 감을 잡을 때 보면 된다.
   - 지금 당장 깊게 팔 필요는 없다.

3. NVFLARE는 production/cross-silo 단계에서 보는 것이 맞다.
   - 현재 TraceMind MVP 단계에서는 우선순위가 낮다.

4. FedML은 예제 참고용 정도로 두는 것이 낫다.

최종 정리로는 아래처럼 답했다.

- 네가 적은 리스트는 70~80% 정도는 맞다.
- 하지만 TraceMind 최종 목표까지 가려면 FL 논문만이 아니라, federated analytics, robust aggregation, time-series 변화 감지, privacy 기초를 같이 공부해야 한다.
- 순서는 `Federated Analytics/계약/통계 -> 로컬 baseline/변화 감지 -> 중앙 aggregation -> FL 이론 -> Federated Reconstruction -> secure aggregation/DP -> Flower/NVFLARE`가 가장 안전하다고 정리했다.
