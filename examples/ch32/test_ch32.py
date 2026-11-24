# 제32장 검증 테스트
#
# 실행: python -m pytest examples/ch32 -q

import struct
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from float_bits import LAYOUT, bits_of, decode  # noqa: E402


def test_fp32_bits_match_struct():
    for x in (0.1, -2.5, 3.0e-5, 12345.678):
        expect = format(struct.unpack(">I", struct.pack(">f", x))[0], "032b")
        assert bits_of(x, torch.float32) == expect


@pytest.mark.parametrize("dt", list(LAYOUT))
def test_decode_matches_dtype_value(dt):
    for x in (0.1, -2.5, 0.0123, 300.0):
        assert decode(bits_of(x, dt), dt) == torch.tensor(x, dtype=dt).item()


def test_layouts_and_finfo():
    for dt, (s, e, m, bias) in LAYOUT.items():
        assert s + e + m == torch.finfo(dt).bits
        assert bias == 2 ** (e - 1) - 1
        assert torch.finfo(dt).eps == 2.0 ** -m


def test_range_vs_precision():
    assert torch.tensor(70000.0, dtype=torch.float16).isinf()
    assert torch.isfinite(torch.tensor(70000.0, dtype=torch.bfloat16))
    assert torch.tensor(2e-8, dtype=torch.float16).item() == 0.0
    w = torch.randn(1000) * 0.02
    err16 = (w.half().float() - w).abs().mean()
    errbf = (w.bfloat16().float() - w).abs().mean()
    assert err16 < errbf                                 # FP16이 더 정밀하다


def test_loss_scaling_recovers_small_gradient():
    g = torch.tensor(2e-8)
    assert g.half().item() == 0.0
    assert (g * 65536).half().float().item() / 65536 == pytest.approx(2e-8, rel=1e-3)
