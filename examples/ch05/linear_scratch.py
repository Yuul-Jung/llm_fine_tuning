# 제05장 핵심 실습 1: Linear Layer를 직접 구현하고 nn.Linear와 비교
#
# 실행: python examples/ch05/linear_scratch.py
#
# 교재 공통 표기 규약: y = x Wᵀ + b,  W의 저장 모양 = [out_features, in_features]

import torch
from torch import nn


def linear_forward(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    """x: [..., in], weight: [out, in], bias: [out] → [..., out]"""
    y = x @ weight.T          # [..., in] × [in, out] = [..., out]
    if bias is not None:
        y = y + bias          # Broadcasting: [out]이 [..., out]으로 확장
    return y


class MyLinear:
    """nn.Linear와 같은 저장 방식으로 만든 최소 Linear 계층."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        # nn.Linear 기본 초기화와 같은 범위: U(-1/√in, 1/√in)
        bound = 1 / in_features ** 0.5
        self.weight = (torch.rand(out_features, in_features) * 2 - 1) * bound
        self.bias = (torch.rand(out_features) * 2 - 1) * bound if bias else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return linear_forward(x, self.weight, self.bias)

    def num_parameters(self) -> int:
        n = self.weight.numel()
        return n + (self.bias.numel() if self.bias is not None else 0)


def main() -> None:
    torch.manual_seed(0)
    layer = nn.Linear(4, 3)        # in 4 → out 3

    print("── 1. nn.Linear의 저장 모양 ──")
    print(f"weight.shape = {tuple(layer.weight.shape)}   ← [out, in]")
    print(f"bias.shape   = {tuple(layer.bias.shape)}")
    params = sum(p.numel() for p in layer.parameters())
    print(f"파라미터 수   = {params}  (4×3 + 3)")

    print("\n── 2. 같은 가중치로 직접 구현과 비교 ──")
    x = torch.randn(2, 4)          # 입력 2개, 각 4차원
    y_torch = layer(x)
    y_mine = linear_forward(x, layer.weight, layer.bias)
    print("nn.Linear :", y_torch.detach())
    print("직접 구현 :", y_mine.detach())
    print(f"최대 절대오차: {(y_torch - y_mine).abs().max().item():.2e}")

    print("\n── 3. 출력 한 칸 손으로 따라가기 ──")
    w0, b0 = layer.weight[0].detach(), layer.bias[0].item()
    manual = sum(x[0, i].item() * w0[i].item() for i in range(4)) + b0
    print(f"y[0, 0] = x[0] · W[0] + b[0] = {manual:.6f}  (출력 {y_torch[0, 0].item():.6f})")

    print("\n── 4. 3차원 입력 [batch, seq, in] ──")
    h = torch.randn(2, 5, 4)
    print(f"{tuple(h.shape)} → {tuple(layer(h).shape)}")

    print("\n── 5. MyLinear(직접 만든 계층) ──")
    mine = MyLinear(4, 3)
    print(f"weight {tuple(mine.weight.shape)}, bias {tuple(mine.bias.shape)}, "
          f"파라미터 {mine.num_parameters()}개, 출력 {tuple(mine(h).shape)}")


if __name__ == "__main__":
    main()
