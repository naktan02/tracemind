# FL Materialized Split Selectors

이 config group은 이미 생성된 FL client split manifest를 실행에서 고르는 축이다.
`fl_data.source_mode=materialized_client_split`, `fl_data.split_manifest`,
`query_data_selection.*`를 함께 고정해 source pair와 manifest path drift를 막는다.

현재 열린 selector는 모두 `shared_client_seed`, 10 clients, Dirichlet alpha 0.3,
seed 42 조합이다.

- `shared_reddit_reddit_pc<N>_alpha03_clients10`: labeled/unlabeled/eval 모두
  `ourafla_reddit`, class당 labeled budget `N`.
- `shared_general_reddit_pc<N>_alpha03_clients10`: labeled는
  `szegeelim_general4`, unlabeled/validation/test는 `ourafla_reddit`,
  class당 labeled budget `N`.

`client_local_labeled` exposure는 현재 main 후보가 아니므로 selector를 만들지 않는다.
