# 제31장 검증 테스트
#
# 실행: python -m pytest examples/ch31 -q

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from num_format import (from_bits, from_fixed, int_range, round_half_away, round_half_even,  # noqa: E402
                        saturate, to_bits, to_fixed, wrap)


def test_ranges():
    assert int_range(8) == (-128, 127) and int_range(4) == (-8, 7)
    assert int_range(8, signed=False) == (0, 255)


@pytest.mark.parametrize("v", [-128, -1, 0, 1, 127])
def test_roundtrip_boundaries(v):
    assert from_bits(to_bits(v, 8)) == v


def test_overflow_raises_and_wraps_like_pytorch():
    with pytest.raises(OverflowError):
        to_bits(128, 8)
    assert wrap(128, 8) == -128 == (torch.tensor([127], dtype=torch.int8) + 1).item()
    assert wrap(-129, 8) == 127


def test_saturation():
    assert saturate(300, 8) == 127 and saturate(-300, 8) == -128 and saturate(5, 8) == 5


def test_fixed_point_q07_and_q34():
    assert to_fixed(0.3, 0, 7) == 38 and from_fixed(38, 7) == 0.296875
    assert to_fixed(1.5, 0, 7) == 127                    # 범위 밖 → 포화
    assert to_fixed(-1.0, 0, 7) == -128                  # 최솟값은 정확히 표현
    assert from_fixed(to_fixed(1.5, 3, 4), 4) == 1.5
    for x in (0.3, -0.71, 0.05):                         # 범위 안의 오차는 간격의 절반 이하
        assert abs(from_fixed(to_fixed(x, 0, 7), 7) - x) <= 0.5 / 128


def test_rounding_modes():
    assert [round_half_even(x) for x in (0.5, 1.5, 2.5, -2.5)] == [0, 2, 2, -2]
    assert [round_half_away(x) for x in (0.5, 1.5, 2.5, -2.5)] == [1, 2, 3, -3]
