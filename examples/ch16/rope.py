# 제16장 핵심 실습: RoPE(회전 위치 임베딩) 직접 구현
#
# 실행: python examples/ch16/rope.py
#
# head_dim 64를 앞 절반(0~31)과 뒤 절반(32~63)으로 나누고, i번째 칸과 (i+32)번째 칸을
# 한 쌍으로 묶어 위치 × 주파수만큼 회전시킨다(Qwen2·Llama 계열의 방식).

import math

import torch


def inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    """쌍마다 회전 속도: θ^(-2i/d), i = 0 … d/2-1. 앞쪽 쌍은 빠르게, 뒤쪽 쌍은 느리게 돈다."""
    return 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)


def rope_cos_sin(seq: int, head_dim: int, theta: float) -> tuple:
    """위치 p, 쌍 i의 회전각 = p × inv_freq[i]. 앞·뒤 절반이 같은 각을 쓰도록 이어 붙인다."""
    angles = torch.outer(torch.arange(seq).float(), inv_freq(head_dim, theta))  # [seq, d/2]
    angles = torch.cat([angles, angles], dim=-1)                                # [seq, d]
    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x₁, x₂] → [-x₂, x₁]: 쌍 (x₁[i], x₂[i])를 90도 돌린 벡터"""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """2차원 회전 공식 x·cosθ + (x를 90도 돌린 것)·sinθ 를 모든 쌍에 한 번에 적용"""
    return x * cos + rotate_half(x) * sin


def rotate_2d(v: torch.Tensor, angle: float) -> torch.Tensor:
    """2차원 벡터 하나를 회전 행렬로 돌린다(비교용)."""
    r = torch.tensor([[math.cos(angle), -math.sin(angle)],
                      [math.sin(angle), math.cos(angle)]])
    return r @ v


def sinusoidal_pe(seq: int, dim: int) -> torch.Tensor:
    """원조 Transformer의 절대 위치 인코딩: 임베딩에 더하는 고정 벡터"""
    pe = torch.zeros(seq, dim)
    pos = torch.arange(seq).float()[:, None]
    freq = 1.0 / 10000 ** (torch.arange(0, dim, 2).float() / dim)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


def score_at(q: torch.Tensor, k: torch.Tensor, m: int, n: int, theta: float = 1e6) -> float:
    """Query를 위치 m, Key를 위치 n에 두었을 때의 RoPE 적용 내적"""
    d = q.shape[-1]
    cos, sin = rope_cos_sin(max(m, n) + 1, d, theta)
    return (apply_rope(q, cos[m], sin[m]) @ apply_rope(k, cos[n], sin[n])).item()


def main() -> None:
    torch.manual_seed(0)

    print("── 1. 위치 정보 없는 Attention은 순서를 모른다 ──")
    x = torch.randn(4, 8)
    perm = torch.tensor([2, 0, 3, 1])
    out = torch.softmax(x @ x.T, dim=-1) @ x
    out_perm = torch.softmax(x[perm] @ x[perm].T, dim=-1) @ x[perm]
    print(f"순서를 섞은 입력의 출력 = 원래 출력을 같은 순서로 섞은 것: {torch.allclose(out[perm], out_perm)}")

    print("\n── 2. 2차원 회전: 길이는 그대로, 방향만 바뀐다 ──")
    v = torch.tensor([3.0, 4.0])
    for deg in (0, 90, 180):
        r = rotate_2d(v, math.radians(deg))
        print(f"  {deg:>3}도 회전: {[round(a, 3) for a in r.tolist()]}  길이 {r.norm().item():.3f}")

    print("\n── 3. 상대 거리만 같으면 점수가 같다(head_dim 64) ──")
    q, k = torch.randn(64), torch.randn(64)
    for m, n in ((3, 1), (10, 8), (100, 98), (10, 1)):
        print(f"  Query 위치 {m:>3}, Key 위치 {n:>3} (거리 {m - n}): 점수 {score_at(q, k, m, n):+.4f}")

    print("\n── 4. 쌍마다 다른 회전 속도(한 바퀴 도는 데 필요한 토큰 수) ──")
    for theta in (1e4, 1e6):
        f = inv_freq(64, theta)
        waves = [2 * math.pi / f[i].item() for i in (0, 8, 16, 31)]
        print(f"  θ = {theta:>9,.0f}: 쌍 0 {waves[0]:.1f}, 쌍 8 {waves[1]:,.0f}, "
              f"쌍 16 {waves[2]:,.0f}, 쌍 31 {waves[3]:,.0f}")

    print("\n── 5. Sinusoidal 위치 인코딩(비교용) ──")
    pe = sinusoidal_pe(16, 8)
    print(f"  모양 {tuple(pe.shape)}: 위치마다 고정 벡터를 임베딩에 더한다. 위치 1의 값:",
          [round(a, 3) for a in pe[1].tolist()])


if __name__ == "__main__":
    main()
