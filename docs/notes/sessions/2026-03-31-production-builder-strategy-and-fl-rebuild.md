# 2026-03-31 production builder strategy와 FL rebuild 요약

이 파일은 archive-only 세션 요약이다. 당시 경로 중 일부는 현재 구조와 다르며,
현재 source of truth는 `methods/prototype/building/*`, `conf/strategy_axes/prototype/*`,
`docs/architecture/system-overview.md`다.

## 핵심 결정

- `single prototype`은 multi-prototype 표현의 특수 케이스로 읽을 수 있게 했다.
- runtime consumer는 category별 prototype list를 처리할 수 있어야 한다.
- production prototype 생성은 실험용 projection/analysis 전략과 분리한다.
- prototype builder core는 scripts가 아니라 `methods/prototype/building/`이 소유한다.
- `single`, `kmeans`, `dbscan` 같은 생성 전략은 한 파일에 몰지 않고 strategy별 파일로 둔다.
- FL rebuild와 seed prototype 생성은 같은 production builder 축을 타야 한다.

## 현재 반영 위치

- builder core: `methods/prototype/building/*`
- builder config: `conf/strategy_axes/prototype/build_strategy/*`
- prototype CLI: `scripts/prototypes/*`
- publication/rebuild runtime: `main_server/src/services/federation/assets/prototypes/*`
- architecture guard: `tests/architecture/test_layer_dependencies.py`

## 남은 판단

- multi-prototype을 active runtime 기본값으로 올릴지, analysis artifact로 유지할지.
- multi-prototype build-state를 exact incremental update까지 지원할지.
