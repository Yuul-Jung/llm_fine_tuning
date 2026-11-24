# 제36장 검증 테스트
#
# 실행: python -m pytest examples/ch36 -q

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch22", HERE.parent / "ch33", HERE.parent / "ch34",
          HERE.parent / "ch35"):
    sys.path.insert(0, str(p))

from int4_pack import dequant_int4, quant_int4  # noqa: E402
from nf4 import CODE, avg_bits, double_dequant, double_quant, nf4_dequant, nf4_fake, nf4_quant  # noqa: E402


def test_codebook_shape():
    assert CODE.numel() == 16 and CODE[0] == -1 and CODE[-1] == 1
    assert (CODE == 0).sum() == 1 and (CODE > 0).sum() == 8 and (CODE < 0).sum() == 7
    assert torch.all(CODE[1:] > CODE[:-1])


def test_matches_bitsandbytes_table():
    bF = pytest.importorskip("bitsandbytes.functional")
    torch.testing.assert_close(bF.get_4bit_type("nf4", device="cpu"), CODE, atol=1e-6, rtol=0)


def test_nearest_codebook_choice():
    idx, absmax = nf4_quant(torch.tensor([0.0, 0.5, -1.0, 0.08] + [0.0] * 60))
    assert absmax.item() == 1.0
    assert CODE[idx[0, :4].long()].tolist() == pytest.approx([0.0, 0.4407, -1.0, 0.0796], abs=1e-4)   # 0.5는 0.4407이 더 가깝다


def test_nf4_beats_int4_on_normal_weights():
    torch.manual_seed(0)
    w = torch.randn(256, 256) * 0.02
    e_int4 = (dequant_int4(*quant_int4(w, 64), 64) - w).norm()
    e_nf4 = (nf4_fake(w) - w).norm()
    assert e_nf4 < 0.9 * e_int4


def test_double_quant_small_error_and_bits():
    torch.manual_seed(1)
    absmax = torch.rand(1024, 1) * 0.1 + 0.02
    back = double_dequant(*double_quant(absmax), absmax.shape)
    assert ((back - absmax).abs() / absmax).max() < 0.01
    assert avg_bits() == 4.5 and avg_bits(dq=True) == pytest.approx(4.127, abs=1e-3)


def test_roundtrip_shape():
    w = torch.randn(8, 128)
    idx, absmax = nf4_quant(w)
    assert idx.dtype == torch.uint8 and idx.max() <= 15
    assert nf4_dequant(idx, absmax, w.shape).shape == w.shape
