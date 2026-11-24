# 제05장 핵심 실습 2: 실제 Qwen2.5-0.5B의 Linear 가중치를 꺼내 직접 계산
#
# 실행: python examples/ch05/qwen_weight.py
# 준비: 처음 실행할 때 모델 파일(model.safetensors, 약 1GB)을 내려받는다.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402

from common.models import qwen_file  # noqa: E402
from linear_scratch import linear_forward  # noqa: E402

LAYER = "model.layers.0.self_attn"


def load_tensors(names: list, local_only: bool = False) -> dict:
    """모델 전체를 올리지 않고, 필요한 텐서만 파일에서 꺼낸다."""
    path = qwen_file("model.safetensors", local_only=local_only)
    with safe_open(path, framework="pt") as f:
        return {name: f.get_tensor(name) for name in names}


def main() -> None:
    names = [f"{LAYER}.k_proj.weight", f"{LAYER}.k_proj.bias", f"{LAYER}.o_proj.weight"]
    t = load_tensors(names)

    print("── 1. 꺼낸 가중치 ──")
    for name, tensor in t.items():
        print(f"{name.replace(LAYER + '.', ''):<16} shape={str(tuple(tensor.shape)):<12} "
              f"dtype={str(tensor.dtype).replace('torch.', '')}")

    w = t[f"{LAYER}.k_proj.weight"].float()      # BF16 저장 → FP32 계산
    b = t[f"{LAYER}.k_proj.bias"].float()
    print(f"\nk_proj 파라미터 수: {w.numel() + b.numel():,}  (896×128 + 128)")

    print("\n── 2. 직접 계산과 PyTorch 함수 비교 ──")
    torch.manual_seed(0)
    x = torch.randn(1, 4, 896)                   # [batch, seq, hidden]
    y_mine = linear_forward(x, w, b)
    y_torch = F.linear(x, w, b)
    print(f"입력 {tuple(x.shape)} → 출력 {tuple(y_mine.shape)}")
    print(f"최대 절대오차: {(y_mine - y_torch).abs().max().item():.2e}")

    print("\n── 3. 가중치 값의 분포 ──")
    print(f"평균 {w.mean().item():+.4f}, 표준편차 {w.std().item():.4f}, "
          f"최솟값 {w.min().item():+.3f}, 최댓값 {w.max().item():+.3f}")


if __name__ == "__main__":
    main()
