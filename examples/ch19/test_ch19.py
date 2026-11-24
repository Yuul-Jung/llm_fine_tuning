# 제19장 검증 테스트
#
# 실행: python -m pytest examples/ch19 -q

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from tiny_model import TINY, Qwen2Tiny, count_by_hand, init_weights  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    return init_weights(Qwen2Tiny(), seed=0).eval()


def test_param_count_matches_hand(tiny):
    hand = count_by_hand()
    assert hand["total"] == 41_914_112
    assert sum(p.numel() for p in tiny.parameters()) == hand["total"]
    assert hand["embedding"] / hand["total"] == pytest.approx(0.928, abs=1e-3)


def test_weight_tying_is_one_tensor(tiny):
    assert tiny.lm_head.weight is tiny.model.embed_tokens.weight
    naive = sum(t.numel() for t in tiny.state_dict().values())
    assert naive - count_by_hand()["total"] == 151_936 * 256


def test_shapes_through_model(tiny):
    trace = []
    with torch.no_grad():
        logits = tiny(torch.tensor([[1, 2, 3]]), trace=trace)
    assert [tuple(s) for _, s in trace[:-1]] == [(1, 3, 256)] * 6
    assert logits.shape == (1, 3, TINY["vocab_size"])


def test_causality(tiny):
    # 뒤 토큰을 바꿔도 앞 위치의 Logits는 그대로여야 한다
    with torch.no_grad():
        a = tiny(torch.tensor([[5, 6, 7, 8]]))
        b = tiny(torch.tensor([[5, 6, 7, 99]]))
    torch.testing.assert_close(a[:, :3], b[:, :3])


def test_hf_qwen2_gives_same_logits(tiny):
    from hf_compat import to_hf
    hf = to_hf(tiny)
    ids = torch.tensor([[9707, 11, 1879, 0, 1234, 42, 7]])
    with torch.no_grad():
        torch.testing.assert_close(tiny(ids), hf(ids).logits, atol=1e-5, rtol=1e-4)
    assert hf.num_parameters() == 41_914_112
