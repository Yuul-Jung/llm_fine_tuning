# 제13장 검증 테스트
#
# 실행: python -m pytest examples/ch13 -q

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attention import K, Q, V, attention, attention_by_hand, softmax_peak  # noqa: E402


def tensors():
    return (torch.tensor(m, dtype=torch.float32) for m in (Q, K, V))


def test_scores_hand_values():
    q, k, _ = tensors()
    assert (q @ k.T)[0].tolist() == [1.0, 0.0, 1.0, 2.0]


def test_first_row_weights_by_hand():
    # 점수 [0.5, 0, 0.5, 1] → e^(s-1) = [e^-0.5, e^-1, e^-0.5, 1]
    e = [math.exp(-0.5), math.exp(-1), math.exp(-0.5), 1.0]
    expected = [x / sum(e) for x in e]
    _, w = attention(*tensors())
    torch.testing.assert_close(w[0], torch.tensor(expected))


def test_three_methods_agree():
    q, k, v = tensors()
    hand, _ = attention_by_hand(Q, K, V)
    mine, _ = attention(q, k, v)
    torch.testing.assert_close(mine, torch.tensor(hand))
    torch.testing.assert_close(mine, F.scaled_dot_product_attention(q, k, v))


def test_weights_rows_sum_to_one():
    _, w = attention(*tensors())
    torch.testing.assert_close(w.sum(dim=-1), torch.ones(4))


def test_output_is_weighted_average_of_values():
    out, w = attention(*tensors())
    torch.testing.assert_close(out, 10 * w)        # V = 10 × 단위행렬


@pytest.mark.parametrize("shape", [(2, 3, 5, 8), (1, 14, 16, 64)])
def test_batched_heads_match_torch(shape):
    torch.manual_seed(0)
    q, k, v = torch.randn(shape), torch.randn(shape), torch.randn(shape)
    torch.testing.assert_close(attention(q, k, v)[0], F.scaled_dot_product_attention(q, k, v))


def test_unscaled_softmax_saturates_with_dimension():
    peaks = [softmax_peak(d, scale=False) for d in (4, 16, 64, 256)]
    assert all(a < b for a, b in zip(peaks, peaks[1:]))
    scaled = [softmax_peak(d) for d in (4, 16, 64, 256)]
    assert max(scaled) - min(scaled) < 0.05
