# 제13장 핵심 실습: Scaled Dot-Product Attention을 손계산·직접 구현·PyTorch로 계산해 비교
#
# 실행: python examples/ch13/attention.py
#
# Attention(Q, K, V) = softmax(Q Kᵀ / √dₖ) V

import math

import torch
import torch.nn.functional as F

# 손계산이 쉽도록 고른 4×4 예제(토큰 4개, dₖ = 4)
Q = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
K = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1]]
V = [[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]]   # 결과 = 10 × 가중치


def attention_by_hand(q, k, v):
    """Python 기본 연산만으로 정의 그대로 계산한다(손계산 재현)."""
    d_k = len(k[0])
    out, all_weights = [], []
    for qi in q:
        scores = [sum(a * b for a, b in zip(qi, kj)) / math.sqrt(d_k) for kj in k]
        exps = [math.exp(s - max(scores)) for s in scores]        # 제06장 안정화
        weights = [e / sum(exps) for e in exps]
        out.append([sum(w * vj[c] for w, vj in zip(weights, v)) for c in range(len(v[0]))])
        all_weights.append(weights)
    return out, all_weights


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: bool = True):
    """텐서 연산으로 구현: [..., seq_q, d] × [..., seq_k, d] → [..., seq_q, d_v]"""
    scores = q @ k.transpose(-2, -1)                       # Q Kᵀ  [..., seq_q, seq_k]
    if scale:
        scores = scores / math.sqrt(q.shape[-1])           # ÷ √dₖ
    weights = torch.softmax(scores, dim=-1)                # 행마다 합 1
    return weights @ v, weights                            # 가중 평균


def softmax_peak(d: int, seq: int = 16, trials: int = 200, scale: bool = True) -> float:
    """무작위 Q·K에서 Attention 가중치 최댓값의 평균: 1에 가까울수록 한 칸에 쏠림."""
    g = torch.Generator().manual_seed(0)
    q = torch.randn(trials, seq, d, generator=g)
    k = torch.randn(trials, seq, d, generator=g)
    _, w = attention(q, k, k, scale=scale)
    return w.max(dim=-1).values.mean().item()


def main() -> None:
    q, k, v = (torch.tensor(m, dtype=torch.float32) for m in (Q, K, V))

    print("── 1. 점수 Q Kᵀ와 ÷√dₖ ──")
    print("Q Kᵀ =\n", q @ k.T)
    print("Q Kᵀ / √4 =\n", q @ k.T / 2)

    print("\n── 2. 세 가지 방법의 결과 ──")
    hand_out, hand_w = attention_by_hand(Q, K, V)
    mine_out, mine_w = attention(q, k, v)
    torch_out = F.scaled_dot_product_attention(q, k, v)
    print("가중치(직접 구현) =\n", mine_w)
    print("행 합:", mine_w.sum(dim=-1).tolist())
    print("출력(직접 구현) =\n", mine_out)
    hand = torch.tensor(hand_out)
    print(f"손계산과 직접 구현 최대 차이: {(hand - mine_out).abs().max().item():.1e}")
    print(f"직접 구현과 PyTorch 최대 차이: {(torch_out - mine_out).abs().max().item():.1e}")

    print("\n── 3. √dₖ로 나누지 않으면: 가중치 최댓값의 평균(토큰 16개) ──")
    print(f"{'dₖ':>6}{'나누지 않음':>14}{'÷√dₖ':>10}")
    for d in (4, 16, 64, 256):
        print(f"{d:>6}{softmax_peak(d, scale=False):>14.3f}{softmax_peak(d):>10.3f}")


if __name__ == "__main__":
    main()
