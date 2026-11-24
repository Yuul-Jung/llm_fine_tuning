# 제27장 실습 1: 작은 행렬로 보는 Rank, SVD, 저랭크 근사
#
# 실행: python examples/ch27/rank_basics.py

import torch


def low_rank(w: torch.Tensor, r: int) -> tuple:
    """SVD로 W ≈ B @ A (B: [out, r], A: [r, in])를 만든다. 특잇값은 B 쪽에 곱해 둔다."""
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    return u[:, :r] * s[:r], vh[:r]


def rel_error(w: torch.Tensor, approx: torch.Tensor) -> float:
    """상대 오차 ‖W − 근사‖ / ‖W‖ (Frobenius 노름)"""
    return ((w - approx).norm() / w.norm()).item()


def main() -> None:
    print("── 1. Rank: 독립된 방향의 수 ──")
    col = torch.tensor([[1.0], [2.0], [3.0]])
    row = torch.tensor([[1.0, 0.0, -1.0, 2.0]])
    r1 = col @ row                                         # 모든 행이 [1, 0, −1, 2]의 배수
    print(f"  열 [1, 2, 3] × 행 [1, 0, −1, 2] =\n{r1}")
    print(f"  원소 {r1.numel()}개, Rank {torch.linalg.matrix_rank(r1).item()}, "
          f"필요한 숫자는 3 + 4 = 7개")
    r2 = r1 + torch.tensor([[0.0], [1.0], [0.0]]) @ torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    print(f"  다른 방향 하나를 더하면 Rank {torch.linalg.matrix_rank(r2).item()}")

    print("\n── 2. SVD: 중요한 방향부터 차례로 ──")
    torch.manual_seed(0)
    w = torch.randn(6, 2) @ torch.randn(2, 5) + 0.05 * torch.randn(6, 5)   # 거의 Rank 2
    s = torch.linalg.svdvals(w)
    print(f"  특잇값 {[round(v, 3) for v in s.tolist()]}")
    for r in (1, 2, 3, 5):
        b, a = low_rank(w, r)
        print(f"  Rank {r}: B {tuple(b.shape)} × A {tuple(a.shape)}, 숫자 {b.numel() + a.numel():>2}개, "
              f"상대 오차 {rel_error(w, b @ a):.4f}")


if __name__ == "__main__":
    main()
