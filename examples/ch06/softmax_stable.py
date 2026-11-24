# 제06장 핵심 실습: 단순 Softmax의 실패와 안정화 Softmax
#
# 실행: python examples/ch06/softmax_stable.py

import torch


def softmax_naive(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """정의 그대로: e^xᵢ / Σ e^xⱼ"""
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)


def softmax_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """최댓값을 먼저 빼고 계산한다. 결과는 수학적으로 같지만 exp가 넘치지 않는다."""
    shifted = x - x.max(dim=dim, keepdim=True).values
    e = torch.exp(shifted)
    return e / e.sum(dim=dim, keepdim=True)


def main() -> None:
    print("── 1. 작은 값: 두 방법 모두 정상 ──")
    small = torch.tensor([1.0, 2.0, 3.0])
    print("단순  :", softmax_naive(small))
    print("안정화:", softmax_stable(small))

    print("\n── 2. 큰 값: 단순 Softmax는 실패 ──")
    large = torch.tensor([1000.0, 1001.0, 1002.0])
    print("exp(1000) =", torch.exp(torch.tensor(1000.0)).item())
    print("단순  :", softmax_naive(large))
    print("안정화:", softmax_stable(large))
    print("torch :", torch.softmax(large, dim=-1))

    print("\n── 3. 최댓값을 빼도 결과가 같은 이유 ──")
    shifted = large - large.max()
    print("[1000, 1001, 1002] - 1002 =", shifted)
    print("[1, 2, 3] - 3             =", small - small.max())
    print("두 입력은 최댓값을 뺀 뒤 같아지므로 Softmax 결과도 같다.")

    print("\n── 4. FP32의 exp가 넘치는 경계 ──")
    for v in (88.0, 88.7, 89.0):
        print(f"exp({v}) = {torch.exp(torch.tensor(v)).item()}")

    print("\n── 5. 실제 어휘 크기(151,936)의 확률분포 ──")
    torch.manual_seed(0)
    logits = torch.randn(151936) * 10
    p = softmax_stable(logits)
    top = torch.topk(p, 3)
    print(f"확률 합: {p.sum().item():.6f},  최대 확률: {top.values[0].item():.4f}")
    print(f"확률이 가장 높은 토큰 ID 3개: {top.indices.tolist()}")


if __name__ == "__main__":
    main()
