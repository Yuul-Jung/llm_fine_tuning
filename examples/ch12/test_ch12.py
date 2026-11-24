# 제12장 검증 테스트
#
# 실행: python -m pytest examples/ch12 -q

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "ch03", ROOT / "ch05", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

from head_split import split_heads  # noqa: E402
from linear_scratch import linear_forward  # noqa: E402


def test_toy_retrieval_prefers_matching_key():
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    values = torch.tensor([[10.0], [20.0], [30.0]])
    weights = torch.softmax(keys @ torch.tensor([0.0, 5.0]), dim=0)
    assert int(weights.argmax()) == 1
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert 20.0 < (weights @ values).item() < 25.0


def test_projection_shapes_small():
    x = torch.randn(2, 5, 32)
    wq, wk = torch.randn(32, 32), torch.randn(8, 32)
    assert linear_forward(x, wq).shape == (2, 5, 32)
    assert linear_forward(x, wk).shape == (2, 5, 8)


@pytest.fixture(scope="module")
def captured():
    try:
        from common.qwen import load_model, load_tokenizer
        model, tok = load_model(True), load_tokenizer(True)
    except Exception:
        pytest.skip("Qwen 모델이 아직 내려받아지지 않음(qkv_projection.py를 먼저 실행)")
    from qkv_projection import TEXT, capture_projections
    ids = tok(TEXT, return_tensors="pt").input_ids
    return model, capture_projections(model, ids), ids.shape[1]


def test_bias_only_on_qkv(captured):
    model, _, _ = captured
    attn = model.model.layers[0].self_attn
    assert all(getattr(attn, n).bias is not None for n in ("q_proj", "k_proj", "v_proj"))
    assert attn.o_proj.bias is None


@pytest.mark.parametrize("name, out_dim", [("q", 896), ("k", 128), ("v", 128)])
def test_direct_equals_hook(captured, name, out_dim):
    model, cap, seq = captured
    lin = getattr(model.model.layers[0].self_attn, f"{name}_proj")
    mine = linear_forward(cap["norm"], lin.weight.detach(), lin.bias.detach())
    assert mine.shape == (1, seq, out_dim)
    torch.testing.assert_close(mine, cap[name], atol=1e-4, rtol=1e-5)


def test_head_shapes(captured):
    _, cap, seq = captured
    assert split_heads(cap["q"], 14).shape == (1, 14, seq, 64)
    assert split_heads(cap["k"], 2).shape == (1, 2, seq, 64)
    assert split_heads(cap["v"], 2).shape == (1, 2, seq, 64)
