# 제35장 검증 테스트
#
# 실행: python -m pytest examples/ch35 -q

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch19", HERE.parent / "ch22", HERE.parent / "ch33",
          HERE.parent / "ch34"):
    sys.path.insert(0, str(p))

from int8_quant import dequant_symmetric, quant_symmetric  # noqa: E402
from quant_error import METHODS, capture_input, fake_quantize_model  # noqa: E402


@pytest.fixture()
def tiny():
    from hf_compat import to_hf
    from tiny_model import Qwen2Tiny, init_weights
    return to_hf(init_weights(Qwen2Tiny(), seed=0)).eval()


def test_error_grows_as_bits_shrink():
    torch.manual_seed(0)
    w, x = torch.randn(128, 256) * 0.05, torch.randn(16, 256)
    errs = [((x @ METHODS[k](w).T - x @ w.T).norm() / (x @ w.T).norm()).item()
            for k in ("INT8 Per-Channel", "INT4 Group 64", "INT4 Per-Channel")]
    assert errs[0] < errs[1] < errs[2]


def test_outlier_inflates_error_of_others():
    torch.manual_seed(1)
    row = torch.randn(1, 512) * 0.05
    base = (dequant_symmetric(*quant_symmetric(row)) - row).abs().mean()
    spiked = row.clone()
    spiked[0, 0] = row.abs().max() * 10
    rest = (dequant_symmetric(*quant_symmetric(spiked)) - spiked)[0, 1:].abs().mean()
    assert rest > 5 * base


def test_capture_input_does_not_change_output(tiny):
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        before = tiny(ids).logits
    x = capture_input(tiny, ids, tiny.model.layers[0].mlp.down_proj)
    with torch.no_grad():
        assert torch.equal(tiny(ids).logits, before)
    assert x.shape == (1, 5, 768)


def test_fake_quantize_touches_all_linears(tiny):
    emb = tiny.model.embed_tokens.weight.detach().clone()
    q_before = tiny.model.layers[2].self_attn.q_proj.weight.detach().clone()
    assert fake_quantize_model(tiny, METHODS["INT8 Per-Channel"]) == 4 * 7
    assert torch.equal(tiny.model.embed_tokens.weight, emb)           # Embedding은 그대로
    q_after = tiny.model.layers[2].self_attn.q_proj.weight
    assert not torch.equal(q_after, q_before)
    assert q_after.abs().amax(dim=1).sub(q_before.abs().amax(dim=1)).abs().max() < 1e-6  # 행 최댓값 보존
