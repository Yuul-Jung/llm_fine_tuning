# 제05장 검증 테스트
#
# 실행: python -m pytest examples/ch05 -q
# Qwen 가중치 테스트는 모델 파일이 이미 내려받아져 있을 때만 실행한다.

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linear_scratch import MyLinear, linear_forward  # noqa: E402


def test_nn_linear_storage_shape_and_count():
    layer = nn.Linear(4, 3)
    assert layer.weight.shape == (3, 4)            # [out, in]
    assert layer.bias.shape == (3,)
    assert sum(p.numel() for p in layer.parameters()) == 15


@pytest.mark.parametrize("shape", [(2, 4), (2, 5, 4), (3, 2, 5, 4)])
def test_linear_forward_matches_nn_linear(shape):
    torch.manual_seed(0)
    layer = nn.Linear(4, 3)
    x = torch.randn(*shape)
    torch.testing.assert_close(linear_forward(x, layer.weight, layer.bias), layer(x))


def test_hand_calculation():
    x = torch.tensor([[1., 2.]])
    w = torch.tensor([[1., 0.], [2., 3.], [0., -1.]])   # [out 3, in 2]
    b = torch.tensor([10., 20., 30.])
    # [1×1+2×0, 1×2+2×3, 1×0+2×(-1)] + b = [11, 28, 28]
    assert torch.equal(linear_forward(x, w, b), torch.tensor([[11., 28., 28.]]))


def test_no_bias():
    w = torch.randn(3, 4)
    x = torch.randn(2, 4)
    torch.testing.assert_close(linear_forward(x, w), F.linear(x, w))


def test_my_linear():
    mine = MyLinear(4, 3)
    assert mine.num_parameters() == 15
    assert mine(torch.randn(2, 5, 4)).shape == (2, 5, 3)
    assert MyLinear(4, 3, bias=False).num_parameters() == 12


def test_qwen_k_proj():
    qwen_weight = pytest.importorskip("qwen_weight")
    names = ["model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.k_proj.bias",
             "model.layers.0.self_attn.o_proj.weight"]
    try:
        t = qwen_weight.load_tensors(names, local_only=True)
    except Exception:
        pytest.skip("Qwen 모델 파일이 아직 내려받아지지 않음(qwen_weight.py를 먼저 실행)")
    w, b = t[names[0]].float(), t[names[1]].float()
    assert w.shape == (128, 896) and b.shape == (128,)
    assert t[names[2]].shape == (896, 896)
    assert t[names[0]].dtype == torch.bfloat16
    x = torch.randn(1, 4, 896)
    torch.testing.assert_close(linear_forward(x, w, b), F.linear(x, w, b))
