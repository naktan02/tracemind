# USB PseudoLabel

이 폴더는 Microsoft Semi-supervised-learning의 `semilearn/algorithms/pseudolabel`
train-step core를 TraceMind Query SSL seam에 맞춰 둔다.

- upstream source: `microsoft/Semi-supervised-learning`
- upstream commit checked: `1ef4cbebcc0b368158315aeb425053858cf6c845`
- 원본 PseudoLabel은 `x_ulb_w`만 사용하므로 TraceMind에서도 strong view를 요구하지 않는다.
- `x_ulb_w`는 query row의 `text` 또는 legacy `weak_text`에서 만든다.
- threshold mask와 masked CE는 `methods/ssl/hooks`의 USB 공통 hook 구현을 재사용한다.
