# 제04장 핵심 실습 1: 내적과 행렬곱을 반복문으로 직접 구현하고 PyTorch와 비교
#
# 실행: python examples/ch04/matmul_loop.py

import time

import torch


def dot(u: torch.Tensor, v: torch.Tensor) -> float:
    """두 벡터의 내적: 같은 위치끼리 곱해 모두 더한다."""
    assert u.shape == v.shape, "내적은 길이가 같은 두 벡터에서만 정의된다"
    total = 0.0
    for i in range(u.shape[0]):
        total += u[i].item() * v[i].item()
    return total


def matmul_loop(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[m, k] × [k, n] = [m, n]: 결과의 (i, j) 칸은 a의 i행과 b의 j열의 내적이다."""
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, f"안쪽 차원이 달라 곱할 수 없음: {tuple(a.shape)} × {tuple(b.shape)}"
    c = torch.zeros(m, n, dtype=a.dtype)
    for i in range(m):
        for j in range(n):
            c[i, j] = dot(a[i, :], b[:, j])
    return c


def main() -> None:
    print("── 1. 손계산 예제 ──")
    a = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.]])          # [2, 3]
    b = torch.tensor([[7., 8.],
                      [9., 10.],
                      [11., 12.]])            # [3, 2]
    print("반복문 결과:\n", matmul_loop(a, b))
    print("torch 결과:\n", a @ b)

    print("\n── 2. 무작위 행렬 비교 ──")
    torch.manual_seed(0)
    a = torch.randn(16, 32)
    b = torch.randn(32, 8)
    loop_c = matmul_loop(a, b)
    torch_c = a @ b
    print(f"모양: {tuple(a.shape)} × {tuple(b.shape)} = {tuple(torch_c.shape)}")
    print(f"최대 절대오차: {(loop_c - torch_c).abs().max().item():.2e}")

    print("\n── 3. 원소별 곱과 행렬곱 ──")
    x = torch.tensor([[1., 2.], [3., 4.]])
    y = torch.tensor([[10., 20.], [30., 40.]])
    print("x * y (원소별 곱):\n", x * y)
    print("x @ y (행렬곱):\n", x @ y)

    print("\n── 4. Broadcasting과 Batch 행렬곱 ──")
    h = torch.randn(2, 4, 8)            # [batch, seq, hidden]
    w = torch.randn(8, 3)               # 모든 문장이 공유하는 가중치
    bias = torch.randn(3)               # [3] → [2, 4, 3]으로 자동 확장
    print(f"[2, 4, 8] @ [8, 3] + [3] = {tuple((h @ w + bias).shape)}")
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    scores = q @ k.transpose(1, 2)      # 문장마다 [4, 8] × [8, 4]
    print(f"Batch 행렬곱 [2, 4, 8] @ [2, 8, 4] = {tuple(scores.shape)}")
    same = torch.allclose(scores[1], q[1] @ k[1].T)
    print(f"묶음 안 문장 1의 결과 = 따로 계산한 결과: {same}")

    print("\n── 5. 속도 비교(64×64) ──")
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    start = time.perf_counter()
    matmul_loop(a, b)
    loop_s = time.perf_counter() - start
    start = time.perf_counter()
    a @ b
    torch_s = time.perf_counter() - start
    print(f"반복문 {loop_s * 1e3:.1f} ms, torch {torch_s * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
