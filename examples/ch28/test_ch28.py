# 제28장 검증 테스트
#
# 실행: python -m pytest examples/ch28 -q

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lora_math import first_step_delta, lora_forward, lora_params, merge  # noqa: E402

W = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
X = torch.tensor([1.0, 2.0, 3.0])
A = torch.tensor([[1.0, 1.0, 0.0]])
B = torch.tensor([[0.5], [-1.0]])


def test_hand_calculation():
    assert lora_forward(X, W, A, B, alpha=2.0, r=1).tolist() == [10.0, -7.0]


def test_merge_equals_lora_forward():
    torch.manual_seed(0)
    w, a, b, x = torch.randn(6, 5), torch.randn(3, 5), torch.randn(6, 3), torch.randn(4, 5)
    torch.testing.assert_close(x @ merge(w, a, b, 16, 3).T, lora_forward(x, w, a, b, 16, 3))


def test_zero_b_keeps_output_and_blocks_a_gradient():
    a = torch.randn(2, 3, requires_grad=True)
    b = torch.zeros(2, 2, requires_grad=True)
    y = lora_forward(X, W, a, b, alpha=8.0, r=2)
    torch.testing.assert_close(y, X @ W.T)
    y.sum().backward()
    assert torch.count_nonzero(a.grad) == 0 and torch.count_nonzero(b.grad) > 0


def test_alpha_acts_like_learning_rate_on_first_step():
    base = first_step_delta(8, 16, 1e-3)
    assert first_step_delta(8, 32, 1e-3) == pytest.approx(2 * base, rel=1e-5)
    assert first_step_delta(8, 16, 2e-3) == pytest.approx(2 * base, rel=1e-5)


def test_parameter_counts():
    assert lora_params(896, 896, 8) == 14_336
    assert lora_params(896, 4864, 8) == 46_080
    assert 24 * (lora_params(896, 896, 8) + lora_params(128, 896, 8)) == 540_672
