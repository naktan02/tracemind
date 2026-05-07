---
name: tracemind-contract-sync
description: Use when changing shared payloads, field meaning, contract interpretation, or any producer-consumer agreement that crosses shared, agent, main_server, scripts, and tests.
---

# TraceMind Contract Sync

이 스킬은 `shared/src/contracts` 또는 그에 준하는 canonical payload shape를
바꿀 때 쓴다.

## 언제 쓸지

- 새 필드를 추가/삭제/이름 변경할 때
- field 의미나 해석 규칙을 바꿀 때
- producer와 consumer가 같은 payload를 다르게 읽고 있을 때
- serialization, manifest, envelope shape를 맞춰야 할 때

## 작업 순서

1. 변경 축을 먼저 적는다.
   - shape 변경인지
   - 의미 변경인지
   - compatibility 추가인지
2. 아래 순서로 읽는다.
   - root `AGENTS.md`
   - `shared/AGENTS.md`
   - `shared/src/contracts/README.md`
   - 관련 contract file
   - 필요한 경우 `docs/contracts/*`
3. contract와 domain 의미를 먼저 고친다.
4. producer와 consumer를 함께 맞춘다.
   - `agent`
   - `main_server`
   - 필요한 경우 `scripts`
5. package test와 integration test를 함께 고친다.
6. README, execution index, plan 중 drift 난 문서를 정리한다.

## Contract Registry 규칙

- canonical shared payload family는 decorator 자동 등록이나 파일 scan으로 발견하지 않는다.
- `shared/src/contracts/**/registry.py`는 payload type 저장, 조회, parse helper 같은 primitive만
  소유한다.
- builtin payload family 연결은 `builtin_loader.py` 같은 명시적 wiring 파일에 둔다.
- compatibility facade는 기존 import path를 유지하는 얇은 표면이어야 한다. payload 의미,
  validation, factory 조합을 facade에 새로 넣지 않는다.
- contract family를 분리할 때 golden fixture를 먼저 두고 load -> parse -> dump shape를
  고정한다.

## 체크리스트

- 같은 의미의 데이터가 경로마다 다른 shape로 흐르지 않는가
- compatibility가 필요하면 핵심 경로와 분리했는가
- 필드 의미가 코드 가까이에 드러나는가
- producer와 consumer가 같은 contract를 보고 있는가
- registry primitive와 builtin wiring이 분리돼 있는가
- facade가 compatibility 외 책임을 흡수하지 않는가
- 테스트와 문서까지 한 흐름으로 닫았는가
