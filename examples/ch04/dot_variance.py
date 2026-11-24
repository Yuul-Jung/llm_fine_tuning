# 제04장 핵심 실습 2: 내적 결과의 분산은 차원 수 d에 비례하는가
#
# 실행: python examples/ch04/dot_variance.py
#
# 가정: 두 벡터의 각 원소가 서로 독립이고, 평균 0·분산 1인 정규분포를 따른다.
# 이론: Var(q·k) = d,  Var(q·k / √d) = 1   → 제13장 Attention의 √dₖ 나누기 근거

import math

import torch

DIMS = [4, 16, 64, 256, 1024]
SAMPLES = 20000


def dot_statistics(d: int, samples: int = SAMPLES, seed: int = 0) -> tuple:
    """길이 d인 무작위 벡터 쌍을 samples개 만들어 내적의 평균과 분산을 구한다."""
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(samples, d, generator=generator)
    k = torch.randn(samples, d, generator=generator)
    dots = (q * k).sum(dim=-1)          # 쌍마다 내적 하나 → [samples]
    return dots.mean().item(), dots.var().item()


def main() -> None:
    print(f"{'d':>6}{'평균':>10}{'분산':>12}{'이론 분산':>10}{'√d로 나눈 뒤 분산':>18}")
    for d in DIMS:
        mean, var = dot_statistics(d)
        # q·k를 √d로 나누면 분산은 1/d배가 된다: Var(X / c) = Var(X) / c²
        print(f"{d:>6}{mean:>10.3f}{var:>12.1f}{d:>10}{var / d:>18.3f}")
    print("\n√d로 나누면 차원과 관계없이 분산이 약 1로 유지된다.")
    print(f"예) d = 64(Qwen2.5-0.5B의 head_dim): 표준편차 √64 = {math.sqrt(64):.0f} → 나눈 뒤 1")


if __name__ == "__main__":
    main()
