# 제15장 실습 1: Multi-Head Attention, GQA, MQA를 같은 함수로 구현하고 비교
#
# 실행: python examples/ch15/mha_gqa.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ch03"))
sys.path.insert(0, str(ROOT / "ch14"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from causal_attention import causal_attention  # noqa: E402  (제14장)
from head_split import merge_heads, split_heads  # noqa: E402  (제03장)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[batch, kv_heads, seq, d] → [batch, kv_heads × n_rep, seq, d]
    KV 헤드 하나를 연속된 Query 헤드 n_rep개가 함께 쓰도록 복제한다."""
    return x.repeat_interleave(n_rep, dim=1)


def grouped_attention(q, k, v, n_heads: int, n_kv: int, rope=None):
    """q: [batch, seq, n_heads×d], k·v: [batch, seq, n_kv×d] → [batch, seq, n_heads×d]"""
    q, k, v = split_heads(q, n_heads), split_heads(k, n_kv), split_heads(v, n_kv)
    if rope is not None:
        q, k = rope(q, k)                          # 위치 회전(제16장)
    k, v = repeat_kv(k, n_heads // n_kv), repeat_kv(v, n_heads // n_kv)
    out, _ = causal_attention(q, k, v)             # 헤드마다 독립 계산
    return merge_heads(out)


def kv_numbers(hidden: int, n_heads: int, n_kv: int, bytes_per: int = 2) -> tuple:
    """K·V Projection 파라미터 수(편향 제외)와 토큰·계층당 KV Cache 크기(FP16, 바이트)."""
    head_dim = hidden // n_heads
    params = 2 * hidden * n_kv * head_dim
    cache = 2 * n_kv * head_dim * bytes_per
    return params, cache


def main() -> None:
    torch.manual_seed(0)
    batch, seq, hidden, n_heads = 1, 6, 16, 4
    x = torch.randn(batch, seq, hidden)
    wq = torch.randn(hidden, hidden)

    print("── 1. 같은 입력, KV 헤드 수만 다르게 ──")
    results = {}
    for name, n_kv in (("MHA", 4), ("GQA", 2), ("MQA", 1)):
        kv_dim = n_kv * hidden // n_heads
        wk, wv = torch.randn(kv_dim, hidden), torch.randn(kv_dim, hidden)
        q, k, v = x @ wq.T, x @ wk.T, x @ wv.T
        out = grouped_attention(q, k, v, n_heads, n_kv)
        ref = F.scaled_dot_product_attention(
            split_heads(q, n_heads), split_heads(k, n_kv), split_heads(v, n_kv),
            is_causal=True, enable_gqa=True)
        error = (out - merge_heads(ref)).abs().max().item()
        results[name] = (k.shape[-1], out.shape)
        print(f"{name}: Query 헤드 {n_heads}, KV 헤드 {n_kv} → K {tuple(k.shape)}, "
              f"출력 {tuple(out.shape)}, PyTorch와 차이 {error:.1e}")

    print("\n── 2. repeat_kv: KV 헤드 2개 → Query 헤드 4개용 ──")
    kv = torch.arange(2).float().view(1, 2, 1, 1).expand(1, 2, 3, 1)   # 헤드 번호를 값으로
    print("복제 전 헤드 번호:", kv[0, :, 0, 0].tolist())
    print("복제 후 헤드 번호:", repeat_kv(kv, 2)[0, :, 0, 0].tolist())

    print("\n── 3. Qwen2.5-0.5B 크기의 K·V 비교(계층 1개, FP16 KV Cache) ──")
    print(f"{'구조':<6}{'KV 헤드':>8}{'K·V 가중치 수':>16}{'토큰당 KV Cache':>18}")
    for name, n_kv in (("MHA", 14), ("GQA", 2), ("MQA", 1)):
        params, cache = kv_numbers(896, 14, n_kv)
        print(f"{name:<6}{n_kv:>8}{params:>16,}{cache:>15,} B")


if __name__ == "__main__":
    main()
