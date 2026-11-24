# 제27장 검증 테스트
#
# 실행: python -m pytest examples/ch27 -q

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch21"):
    sys.path.insert(0, str(p))

from rank_basics import low_rank, rel_error  # noqa: E402


def test_outer_product_is_rank_one():
    m = torch.tensor([[1.0], [2.0], [3.0]]) @ torch.tensor([[1.0, 0.0, -1.0, 2.0]])
    assert torch.linalg.matrix_rank(m).item() == 1


def test_full_rank_reconstructs_exactly():
    torch.manual_seed(0)
    w = torch.randn(7, 5)
    b, a = low_rank(w, 5)
    assert b.shape == (7, 5) and a.shape == (5, 5)
    torch.testing.assert_close(b @ a, w, atol=1e-5, rtol=1e-5)


def test_error_equals_dropped_singular_values():
    # Eckart–Young: 잘린 특잇값의 에너지가 곧 오차
    torch.manual_seed(1)
    w = torch.randn(20, 12)
    s = torch.linalg.svdvals(w)
    for r in (1, 4, 8):
        b, a = low_rank(w, r)
        expect = (s[r:].pow(2).sum() / s.pow(2).sum()).sqrt().item()
        assert rel_error(w, b @ a) == pytest.approx(expect, rel=1e-4)


def test_parameter_saving():
    out_dim, in_dim, r = 896, 896, 8
    assert r * (out_dim + in_dim) == 14_336
    assert r * (out_dim + in_dim) / (out_dim * in_dim) == pytest.approx(0.01786, abs=1e-5)


def test_qwen_error_decreases_with_rank():
    try:
        from common.qwen import load_model
        model = load_model(True)
    except Exception:
        pytest.skip("Qwen 모델이 아직 내려받아지지 않음")
    from qwen_svd import NAME, RANKS, error_curve
    curve = error_curve(dict(model.named_parameters())[NAME].detach())
    assert all(a > b for a, b in zip(curve, curve[1:]))
    assert curve[-1] < 1e-4
