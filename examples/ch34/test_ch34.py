# 제34장 검증 테스트
#
# 실행: python -m pytest examples/ch34 -q

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch33"):
    sys.path.insert(0, str(p))

from int4_pack import dequant_int4, pack, quant_int4, sign_extend4, unpack  # noqa: E402


def test_hand_byte():
    assert int(pack(torch.tensor([3, -2]))) == 0xE3


def test_sign_extension_all_nibbles():
    v = torch.arange(16)
    assert sign_extend4(v).tolist() == list(range(8)) + list(range(-8, 0))


def test_roundtrip_random():
    torch.manual_seed(0)
    q = torch.randint(-8, 8, (64, 32), dtype=torch.int8)
    p = pack(q)
    assert p.dtype == torch.uint8 and p.numel() == q.numel() // 2
    assert torch.equal(unpack(p, q.shape), q)


def test_int4_error_bound_and_groups():
    torch.manual_seed(1)
    w = torch.randn(16, 128) * 0.05
    for g in (None, 64, 32):
        q, s = quant_int4(w, g)
        assert q.abs().max() <= 7
        err = (dequant_int4(q, s, g) - w).reshape(-1, g or 128).abs()
        assert (err <= s / 2 + 1e-7).all()
    e = [((dequant_int4(*quant_int4(w, g), g) - w).norm()).item() for g in (None, 64, 32)]
    assert e[0] > e[1] > e[2]                        # 묶음이 작을수록 오차 감소


def test_odd_length_rejected():
    with pytest.raises(AssertionError):
        pack(torch.tensor([1, 2, 3]))
