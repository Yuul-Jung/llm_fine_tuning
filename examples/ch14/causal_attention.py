# 제14장 실습 1: Causal Mask와 Padding Mask를 포함한 Self-Attention 직접 구현
#
# 실행: python examples/ch14/causal_attention.py

import math

import torch
import torch.nn.functional as F


def causal_mask(seq: int) -> torch.Tensor:
    """True = 가림. 자기보다 뒤(미래) 위치를 가리는 상삼각 행렬."""
    return torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)


def causal_attention(q, k, v, padding_mask=None):
    """q, k, v: [batch, heads, seq, d],  padding_mask: [batch, seq] (1 = 실제 토큰, 0 = PAD)"""
    seq = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])      # [batch, heads, seq, seq]
    blocked = causal_mask(seq)                                     # [seq, seq]
    if padding_mask is not None:
        pad = padding_mask[:, None, None, :] == 0                  # [batch, 1, 1, seq] (제03장)
        blocked = blocked | pad                                    # 두 마스크 결합(Broadcasting)
    scores = scores.masked_fill(blocked, float("-inf"))            # 가린 칸 = -∞
    weights = torch.softmax(scores, dim=-1)                        # e^(-∞) = 0
    return weights @ v, weights


def main() -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 4, 8) for _ in range(3))

    print("── 1. Causal Mask(True = 가림) ──")
    print(causal_mask(4).int())

    print("\n── 2. 마스크를 적용한 가중치 ──")
    out, w = causal_attention(q, k, v)
    print(w[0, 0].round(decimals=3))
    print("행 합:", [round(x, 4) for x in w[0, 0].sum(dim=-1).tolist()])
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"PyTorch(is_causal=True)와 최대 차이: {(out - ref).abs().max().item():.1e}")

    print("\n── 3. 마지막 토큰을 바꾸면? ──")
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    for t in (q2, k2, v2):
        t[0, 0, -1] = torch.randn(8)
    out2, _ = causal_attention(q2, k2, v2)
    print(f"앞 3개 위치 출력 완전 일치: {torch.equal(out[0, 0, :3], out2[0, 0, :3])}")
    print(f"마지막 위치 출력 차이: {(out[0, 0, 3] - out2[0, 0, 3]).abs().max().item():.3f}")

    print("\n── 4. Padding Mask 결합(두 번째 문장은 실제 토큰 2개 + PAD 2개) ──")
    qb, kb, vb = (torch.randn(2, 1, 4, 8) for _ in range(3))
    padding = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    _, wb = causal_attention(qb, kb, vb, padding_mask=padding)
    print(wb[1, 0].round(decimals=3))
    print(f"PAD 위치(열 2·3)로 가는 가중치 합: {wb[1, 0, :, 2:].sum().item():.1f}")


if __name__ == "__main__":
    main()
