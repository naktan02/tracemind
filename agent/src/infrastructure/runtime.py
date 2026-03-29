"""런타임 장치(device) 해석 유틸리티."""

from __future__ import annotations


def resolve_runtime_device(device: str = "auto") -> str:
    """문자열 장치 지정을 실제 사용 가능한 장치로 정규화한다.

    'auto'이면 CUDA가 보일 때 'cuda', 아니면 'cpu'를 반환한다.
    명시 지정('cuda', 'cuda:0', 'cpu', 'mps' 등)은 그대로 통과시킨다.
    """
    if device != "auto":
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    return "cpu"
