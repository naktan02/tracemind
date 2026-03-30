  - 계약부터 올립니다. shared/src/contracts/prototype_contracts.py:12
      - 지금: category -> centroid 1개
      - 변경: category -> prototypes[]
      - single은 길이 1인 리스트로 표현
  - scoring을 바꿉니다. agent/src/services/inference/scoring_service.py:16
      - 지금: 카테고리당 vector 1개와 cosine
      - 변경: 카테고리 내 여러 prototype 점수 중 max 또는 log-sum-exp
  - runtime service를 바꿉니다. agent/src/services/prototype/runtime_service.py:18
      - 지금 get_active_centroids()
      - 변경 get_active_prototypes() 추가, 기존 centroids alias는 single 호환용으로만 유지
  - builder를 바꿉니다. scripts/prototypes/prototype_pack_builder.py:33
      - 지금: 카테고리 평균 centroid 1개
      - 변경: builder가 list[prototype]를 만들게
      - single builder는 list 길이 1 반환
  - simulation/evaluation을 바꿉니다. scripts/experiments/run_federated_simulation.py:398
      - extract_category_centroids() 대신 multi 구조를 읽게
      - 나머지 shared adapter FL 루프는 대부분 유지 가능

  즉 로컬 update/aggregation 쪽보다 runtime/prototype contract 쪽이 주 변경점입니다.