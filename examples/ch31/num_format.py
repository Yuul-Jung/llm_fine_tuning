# 제31장 핵심 실습: Python 비트 연산으로 만드는 정수·고정소수점 변환기
#
# 실행: python examples/ch31/num_format.py

import torch


def int_range(bits: int, signed: bool = True) -> tuple:
    """n비트 정수의 (최솟값, 최댓값). 부호 있는 정수는 2의 보수."""
    return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)


def to_bits(value: int, bits: int) -> str:
    """정수를 n비트 2의 보수 비트열로. 음수는 2ⁿ을 더한 것과 같다."""
    lo, hi = int_range(bits)
    if not lo <= value <= hi:
        raise OverflowError(f"{value}는 {bits}비트 범위 [{lo}, {hi}] 밖")
    return format(value & ((1 << bits) - 1), f"0{bits}b")


def from_bits(pattern: str, signed: bool = True) -> int:
    """비트열을 정수로. 부호 있는 정수는 맨 앞 비트가 −2ⁿ⁻¹의 자리값을 가진다."""
    u = int(pattern, 2)
    return u - (1 << len(pattern)) if signed and pattern[0] == "1" else u


def wrap(value: int, bits: int) -> int:
    """Overflow: 범위를 넘으면 반대쪽 끝으로 돌아간다(정수 하드웨어의 기본 동작)."""
    return from_bits(format(value & ((1 << bits) - 1), f"0{bits}b"))


def saturate(value: int, bits: int) -> int:
    """Saturation: 범위를 넘으면 가장 가까운 끝값에 머문다."""
    lo, hi = int_range(bits)
    return max(lo, min(hi, value))


def round_half_even(x: float) -> int:
    """반올림(짝수 쪽으로): 정확히 .5일 때 짝수를 고른다. Python round()와 IEEE 754 기본값."""
    return round(x)


def round_half_away(x: float) -> int:
    """반올림(0에서 먼 쪽으로): 학교에서 배우는 사사오입"""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def to_fixed(x: float, int_bits: int, frac_bits: int) -> int:
    """실수 → Qm.n 고정소수점 정수 코드(전체 비트 = 1 + m + n, 반올림 후 Saturation)"""
    return saturate(round_half_even(x * (1 << frac_bits)), 1 + int_bits + frac_bits)


def from_fixed(code: int, frac_bits: int) -> float:
    return code / (1 << frac_bits)


def main() -> None:
    print("── 1. 같은 8비트, 다른 해석 ──")
    for pattern in ("00000101", "01111111", "10000000", "11111111"):
        print(f"  {pattern}: 부호 없음 {from_bits(pattern, False):>4}, 부호 있음(2의 보수) {from_bits(pattern):>5}")
    print(f"  −5 = {to_bits(-5, 8)}  (5 = {to_bits(5, 8)}을 뒤집고 1 더하기)")

    print("\n── 2. 비트 폭과 표현 범위 ──")
    for bits in (4, 8, 16, 32):
        lo, hi = int_range(bits)
        print(f"  INT{bits:<3}[{lo:>14,}, {hi:>13,}]  값 {1 << bits:,}가지")

    print("\n── 3. Q Format 고정소수점(8비트) ──")
    for m, n in ((0, 7), (3, 4)):
        step = 1 / (1 << n)
        lo, hi = int_range(1 + m + n)
        print(f"  Q{m}.{n}: 간격 {step:g}, 범위 [{from_fixed(lo, n)}, {from_fixed(hi, n)}]")
        for x in (0.3, -0.71, 1.5):
            code = to_fixed(x, m, n)
            print(f"    {x:>5} → 코드 {code:>4} ({to_bits(code, 8)}) → {from_fixed(code, n):>9.6f}, "
                  f"오차 {abs(from_fixed(code, n) - x):.5f}")

    print("\n── 4. Overflow, Saturation, Rounding ──")
    print(f"  INT8 127 + 1: 감싸기 {wrap(128, 8)}, 포화 {saturate(128, 8)}")
    t = torch.tensor([127], dtype=torch.int8)
    print(f"  PyTorch int8 텐서 127 + 1 = {(t + 1).item()}  (오류 없이 감싸기)")
    for x in (2.5, 3.5, -2.5, 2.4999):
        print(f"  {x:>7}: 짝수 쪽 {round_half_even(x):>3}, 0에서 먼 쪽 {round_half_away(x):>3}")


if __name__ == "__main__":
    main()
