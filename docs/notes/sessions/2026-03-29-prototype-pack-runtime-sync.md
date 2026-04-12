# 2026-03-29 Prototype Pack Runtime Sync

## User

다음 순서까지는 말고 지금 순서부터 진행 시작해
그리고 궁금한게 centroid는 무슨 수학적 수식으로 계산해서 결과 나온거야?
그리고 skills에도 추가하고 너도 기억해야 할점은 선택이 필요한 순간에는 무조건 나에게 말하고 내 의견을 확인하고 진행해
그리고 그럼 이제 prototype_packs에 저장할 때 메인서버에도 동시에 저장하는걸로 진행하자

## Assistant

- `PrototypePack` 파일 기반 배포 경로를 현재 단계의 우선 작업으로 잡고 구현을 시작했다.
- 공용 `PrototypePack` 전송 contract와 직렬화 유틸리티를 `shared/src/contracts/prototype_contracts.py`에 추가했다.
- 중앙 서버에는 `main-server/state/prototype_packs/versions/`와 `active.json` 기반 file store를 도입하고, publish/activate/current 조회용 service와 API를 추가했다.
- agent에는 `agent/state/prototype_packs/versions/`와 `active.json` 기반 로컬 캐시를 도입하고, 현재 활성 pack 조회 및 중앙에서 pull하는 최소 sync 경로를 추가했다.
- `scripts/seed_prototypes.py`는 local `data/processed/prototype_packs`에 저장할 때 메인서버 state 저장소에도 동일한 pack을 동시에 publish하도록 수정했다.
- 자동 activate는 사용자가 명시적으로 선택하지 않았으므로 기본 동작에 넣지 않았고, 대신 `scripts/activate_prototype_pack.py`를 추가해 수동 활성화 경로를 분리했다.
- `scripts/pull_prototype_pack.py`도 추가해서 agent 쪽 로컬 캐시 pull을 스크립트로 수행할 수 있게 했다.
- 사용자 선호인 “선택이 필요한 순간에는 사용자 확인 후 진행” 규칙은 로컬 memory 파일에 기록했다. 전역 skill 파일 자체는 워크스페이스 밖이라 이번 저장소 패치 범위에는 포함하지 않았다.
- 현재 이미 생성된 ourafla pack은 `main-server/state/prototype_packs/versions/ourafla_mxbai_20260328T153818Z.json`로 publish해 두었다.
- centroid 계산식은 카테고리 내 임베딩들의 좌표별 산술평균이며, 이후 query 점수는 해당 centroid와의 cosine similarity로 계산한다.
