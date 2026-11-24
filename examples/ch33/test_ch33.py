# 제33장 검증 테스트
#
# 실행: python -m pytest examples/ch33 -q

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from int8_quant import dequant_asymmetric, dequant_symmetric, quant_asymmetric, quant_symmetric  # noqa: E402


def test_hand_example():
    q, s = quant_symmetric(torch.tensor([-0.5, 0.0, 0.25, 1.0]))
    assert q.tolist() == [-64, 0, 32, 127] and q.dtype == torch.int8
    qa, sa, za = quant_asymmetric(torch.tensor([-0.5, 0.0, 0.25, 1.0]))
    assert qa.tolist() == [0, 85, 127, 255] and int(za) == 85
    assert dequant_asymmetric(qa, sa, za)[1].item() == 0.0      # 0이 정확히 표현된다


@pytest.mark.parametrize("pc", [False, True])
def test_error_within_half_scale(pc):
    torch.manual_seed(0)
    w = torch.randn(64, 48) * torch.linspace(0.01, 1.0, 64)[:, None]
    q, s = quant_symmetric(w, per_channel=pc)
    err = (dequant_symmetric(q, s) - w).abs()
    assert (err <= s.expand_as(w) / 2 + 1e-7).all() if pc else (err <= s / 2 + 1e-7).all()


def test_per_channel_beats_per_tensor_when_rows_differ():
    torch.manual_seed(1)
    w = torch.randn(32, 32) * torch.logspace(-3, 0, 32)[:, None]   # 행마다 크기가 크게 다름
    e_t = (dequant_symmetric(*quant_symmetric(w)) - w).norm()
    e_c = (dequant_symmetric(*quant_symmetric(w, per_channel=True)) - w).norm()
    assert e_c < e_t / 2


def test_floor_is_biased_round_is_not():
    torch.manual_seed(2)
    w = torch.randn(10000) * 0.1
    up = dequant_symmetric(*quant_symmetric(w)) - w
    down = dequant_symmetric(*quant_symmetric(w, rounding=torch.floor)) - w
    assert abs(up.mean()) < 1e-4 and down.mean() < -1e-4


def test_qwen_storage_is_quarter():
    try:
        from common.qwen import load_model
        model = load_model(True)
    except Exception:
        pytest.skip("Qwen 모델이 아직 내려받아지지 않음")
    from int8_quant import NAME
    w = dict(model.named_parameters())[NAME].detach()
    q, s = quant_symmetric(w, per_channel=True)
    assert q.numel() * q.element_size() + s.numel() * 4 == w.numel() + 896 * 4
    assert ((dequant_symmetric(q, s) - w).norm() / w.norm()).item() < 0.02
