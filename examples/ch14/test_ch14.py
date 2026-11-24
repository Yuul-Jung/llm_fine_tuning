# 제14장 검증 테스트
#
# 실행: python -m pytest examples/ch14 -q

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_attention import causal_attention, causal_mask  # noqa: E402


def test_mask_shape_and_pattern():
    m = causal_mask(4)
    assert m.shape == (4, 4)
    assert m.sum().item() == 6                    # 상삼각 칸 수 = 4 × 3 / 2
    assert not m.diagonal().any()                 # 자기 자신은 볼 수 있다


def test_weights_lower_triangular_and_rows_sum_to_one():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 5, 8) for _ in range(3))
    _, w = causal_attention(q, k, v)
    assert torch.all(w.masked_select(causal_mask(5)) == 0)
    torch.testing.assert_close(w.sum(dim=-1), torch.ones(2, 3, 5))


def test_matches_pytorch_is_causal():
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 14, 16, 64) for _ in range(3))
    torch.testing.assert_close(causal_attention(q, k, v)[0],
                               F.scaled_dot_product_attention(q, k, v, is_causal=True))


def test_future_change_does_not_affect_past():
    torch.manual_seed(2)
    q, k, v = (torch.randn(1, 1, 6, 8) for _ in range(3))
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    for t in (q2, k2, v2):
        t[0, 0, -1] = torch.randn(8)
    assert torch.equal(causal_attention(q, k, v)[0][..., :-1, :],
                       causal_attention(q2, k2, v2)[0][..., :-1, :])


def test_padding_columns_get_zero_weight():
    torch.manual_seed(3)
    q, k, v = (torch.randn(2, 1, 4, 8) for _ in range(3))
    _, w = causal_attention(q, k, v, padding_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]))
    assert w[1, 0, :, 2:].sum().item() == 0.0
    assert w[0, 0, 3, 3].item() > 0              # PAD가 없는 문장은 그대로


@pytest.fixture(scope="module")
def qwen():
    try:
        from common.qwen import load_model, load_tokenizer
        return load_model(True), load_tokenizer(True)
    except Exception:
        pytest.skip("Qwen 모델이 아직 내려받아지지 않음(qwen_causal_check.py를 먼저 실행)")


def test_qwen_future_change(qwen):
    from qwen_causal_check import future_change_test
    past, last = future_change_test(*qwen)
    assert past == 0.0 and last > 1.0


def test_qwen_padding_matches_alone(qwen):
    from qwen_causal_check import padding_test
    batch, diff = padding_test(*qwen)
    assert qwen[1].padding_side == "right"
    assert batch.attention_mask[1].tolist()[:4] == [1, 1, 1, 0]
    assert diff < 1e-3
