# 제03장 핵심 실습: Attention 헤드 분할과 결합(모양 바꾸기의 왕복 검증)
#
# 실행: python examples/ch03/head_split.py
#
# [batch, seq, hidden] → [batch, heads, seq, head_dim] → [batch, seq, hidden]
# 제15장 Multi-Head Attention에서 이 두 함수를 그대로 다시 사용한다.

import torch


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """hidden 축을 num_heads개의 헤드로 나누고, 헤드 축을 seq 앞으로 옮긴다."""
    batch, seq, hidden = x.shape
    head_dim = hidden // num_heads
    x = x.view(batch, seq, num_heads, head_dim)     # 1) hidden → (heads, head_dim)
    return x.transpose(1, 2)                         # 2) seq ↔ heads 축 교환


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """split_heads의 역연산: 헤드 축을 되돌리고 (heads, head_dim)을 hidden으로 합친다."""
    batch, heads, seq, head_dim = x.shape
    x = x.transpose(1, 2)                            # 1) heads ↔ seq 축 되돌리기
    return x.reshape(batch, seq, heads * head_dim)   # 2) 연속 저장이 아니므로 reshape 사용


def main() -> None:
    batch, seq, hidden, heads = 2, 4, 8, 2
    x = torch.arange(batch * seq * hidden, dtype=torch.float32).reshape(batch, seq, hidden)

    print("단계별 모양 추적")
    print(f"  입력 x                 : {tuple(x.shape)}   예측 (2, 4, 8)")
    step1 = x.view(batch, seq, heads, hidden // heads)
    print(f"  1) view로 hidden 분할  : {tuple(step1.shape)} 예측 (2, 4, 2, 4)")
    y = split_heads(x, heads)
    print(f"  2) transpose(1, 2)     : {tuple(y.shape)} 예측 (2, 2, 4, 4)")
    z = merge_heads(y)
    print(f"  3) 되돌리기            : {tuple(z.shape)}   예측 (2, 4, 8)")

    print(f"\n왕복 후 원본과 완전 일치: {torch.equal(x, z)}")

    # 원소 하나의 이동 확인: x[b, s, h*head_dim + d] == y[b, h, s, d]
    b, s, h, d = 1, 2, 1, 3
    head_dim = hidden // heads
    print(f"x[{b}, {s}, {h * head_dim + d}] = {x[b, s, h * head_dim + d].item():.0f}  "
          f"→  y[{b}, {h}, {s}, {d}] = {y[b, h, s, d].item():.0f}")

    print("\n헤드 0이 보는 데이터(문장 0): 각 토큰의 hidden 앞쪽 4칸")
    print(y[0, 0])

    # 실제 Qwen2.5-0.5B 크기로 같은 함수 적용: hidden 896, Query 헤드 14
    q = torch.randn(1, 16, 896)
    print(f"\nQwen2.5-0.5B 크기: {tuple(q.shape)} → {tuple(split_heads(q, 14).shape)}")


if __name__ == "__main__":
    main()
