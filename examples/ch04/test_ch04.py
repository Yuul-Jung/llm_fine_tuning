# 제04장 검증 테스트
#
# 실행: python -m pytest examples/ch04 -q

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dot_variance import dot_statistics  # noqa: E402
from matmul_loop import dot, matmul_loop  # noqa: E402


def test_dot_hand_calculation():
    # 1×4 + 2×5 + 3×6 = 32
    assert dot(torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.])) == 32.0


def test_matmul_hand_calculation():
    a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    b = torch.tensor([[7., 8.], [9., 10.], [11., 12.]])
    expected = torch.tensor([[58., 64.], [139., 154.]])
    assert torch.equal(matmul_loop(a, b), expected)


def test_matmul_matches_torch():
    torch.manual_seed(0)
    a, b = torch.randn(16, 32), torch.randn(32, 8)
    torch.testing.assert_close(matmul_loop(a, b), a @ b, atol=1e-5, rtol=1e-5)


def test_inner_dimension_mismatch():
    with pytest.raises(AssertionError):
        matmul_loop(torch.zeros(2, 3), torch.zeros(2, 3))


def test_hadamard_is_not_matmul():
    x = torch.tensor([[1., 2.], [3., 4.]])
    y = torch.tensor([[10., 20.], [30., 40.]])
    assert torch.equal(x * y, torch.tensor([[10., 40.], [90., 160.]]))
    assert torch.equal(x @ y, torch.tensor([[70., 100.], [150., 220.]]))


def test_broadcast_and_batch_shapes():
    assert (torch.randn(2, 4, 8) @ torch.randn(8, 3) + torch.randn(3)).shape == (2, 4, 3)
    q, k = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    scores = q @ k.transpose(1, 2)
    assert scores.shape == (2, 4, 4)
    torch.testing.assert_close(scores[1], q[1] @ k[1].T)


@pytest.mark.parametrize("d", [16, 64, 256])
def test_dot_variance_close_to_d(d):
    # 표본 20,000개의 분산 추정 오차를 고려해 이론값 d의 ±5% 이내를 통과로 본다.
    mean, var = dot_statistics(d)
    assert abs(mean) < 0.1 * d ** 0.5
    assert abs(var / d - 1) < 0.05
