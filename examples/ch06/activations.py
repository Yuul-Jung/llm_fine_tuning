# 제06장 실습 1: 비선형이 필요한 이유와 활성화 함수 직접 구현
#
# 실행: python examples/ch06/activations.py
# 결과: 화면 비교표 + examples/ch06/activations.png(matplotlib이 있으면)

import math
from pathlib import Path

import torch
import torch.nn.functional as F


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """0과 1 사이로 눌러 담는다: 1 / (1 + e^(-x))"""
    return 1 / (1 + torch.exp(-x))


def relu(x: torch.Tensor) -> torch.Tensor:
    """음수는 0, 양수는 그대로: max(0, x)"""
    return torch.clamp(x, min=0)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """x에 '정규분포에서 x 이하일 확률'을 곱한다: x · Φ(x)"""
    return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def silu(x: torch.Tensor) -> torch.Tensor:
    """x에 자기 자신의 sigmoid를 곱한다: x · σ(x)  (Qwen2.5가 쓰는 활성화 함수)"""
    return x * sigmoid(x)


def two_linears(x, w1, b1, w2, b2, activation=None):
    """Linear → (활성화) → Linear"""
    h = x @ w1.T + b1
    if activation is not None:
        h = activation(h)
    return h @ w2.T + b2


def main() -> None:
    print("── 1. Linear 두 개는 Linear 하나와 같다 ──")
    torch.manual_seed(0)
    w1, b1 = torch.randn(5, 3), torch.randn(5)
    w2, b2 = torch.randn(2, 5), torch.randn(2)
    x = torch.randn(4, 3)
    # 합친 가중치: W = W2·W1,  합친 편향: b = W2·b1 + b2
    w, b = w2 @ w1, w2 @ b1 + b2
    stacked = two_linears(x, w1, b1, w2, b2)
    merged = x @ w.T + b
    print(f"두 층 결과와 한 층 결과의 최대 차이: {(stacked - merged).abs().max().item():.2e}")
    with_relu = two_linears(x, w1, b1, w2, b2, activation=relu)
    print(f"사이에 ReLU를 넣으면 최대 차이: {(with_relu - merged).abs().max().item():.2f}")

    print("\n── 2. 활성화 함수 값 비교 ──")
    points = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    table = {
        "Sigmoid": (sigmoid, torch.sigmoid),
        "ReLU": (relu, F.relu),
        "GELU": (gelu, F.gelu),
        "SiLU": (silu, F.silu),
    }
    print(f"{'x':>8}" + "".join(f"{v:>9.1f}" for v in points.tolist()))
    for name, (mine, ref) in table.items():
        values = mine(points)
        error = (values - ref(points)).abs().max().item()
        print(f"{name:>8}" + "".join(f"{v:>9.4f}" for v in values.tolist())
              + f"   (PyTorch와 최대 차이 {error:.1e})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    xs = torch.linspace(-4, 4, 200)
    for name, (mine, _) in table.items():
        plt.plot(xs, mine(xs), label=name)
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = Path(__file__).with_name("activations.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n그래프 저장: {out.name}")


if __name__ == "__main__":
    main()
