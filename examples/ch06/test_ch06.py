# 제06장 검증 테스트
#
# 실행: python -m pytest examples/ch06 -q

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from activations import gelu, relu, sigmoid, silu, two_linears  # noqa: E402
from softmax_stable import softmax_naive, softmax_stable  # noqa: E402


@pytest.mark.parametrize("mine, ref", [
    (sigmoid, torch.sigmoid), (relu, F.relu), (gelu, F.gelu), (silu, F.silu)])
def test_activation_matches_torch(mine, ref):
    x = torch.linspace(-6, 6, 101)
    torch.testing.assert_close(mine(x), ref(x))


def test_activation_hand_values():
    assert sigmoid(torch.tensor(0.0)).item() == 0.5
    assert relu(torch.tensor(-2.0)).item() == 0.0
    assert silu(torch.tensor(0.0)).item() == 0.0


def test_linear_stack_collapses_without_activation():
    torch.manual_seed(0)
    w1, b1, w2, b2 = torch.randn(5, 3), torch.randn(5), torch.randn(2, 5), torch.randn(2)
    x = torch.randn(4, 3)
    merged = x @ (w2 @ w1).T + (w2 @ b1 + b2)
    torch.testing.assert_close(two_linears(x, w1, b1, w2, b2), merged)
    assert not torch.allclose(two_linears(x, w1, b1, w2, b2, activation=relu), merged)


def test_softmax_sums_to_one_and_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(3, 10) * 5
    p = softmax_stable(x)
    torch.testing.assert_close(p.sum(dim=-1), torch.ones(3))
    torch.testing.assert_close(p, torch.softmax(x, dim=-1))


def test_naive_fails_stable_succeeds_on_large_logits():
    large = torch.tensor([1000.0, 1001.0, 1002.0])
    assert torch.isnan(softmax_naive(large)).any()
    torch.testing.assert_close(softmax_stable(large), torch.softmax(large, dim=-1))


def test_shift_invariance():
    x = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(softmax_stable(x), softmax_stable(x + 1000))
