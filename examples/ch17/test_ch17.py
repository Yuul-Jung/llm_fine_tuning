# 제17장 검증 테스트
#
# 실행: python -m pytest examples/ch17 -q

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "ch05", ROOT / "ch06", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

from swiglu import swiglu_ffn  # noqa: E402


def test_hand_example():
    # hidden 2, intermediate 2: gate·up·down을 단위행렬로 두면 FFN(x) = silu(x) ⊙ x
    eye = torch.eye(2)
    x = torch.tensor([[1.0, -2.0]])
    expected = F.silu(x) * x
    torch.testing.assert_close(swiglu_ffn(x, eye, eye, eye), expected)


def test_shapes_and_tokenwise():
    torch.manual_seed(0)
    wg, wu, wd = torch.randn(12, 4), torch.randn(12, 4), torch.randn(4, 12)
    x = torch.randn(2, 5, 4)
    y = swiglu_ffn(x, wg, wu, wd)
    assert y.shape == (2, 5, 4)
    torch.testing.assert_close(y[:, 3], swiglu_ffn(x[:, 3], wg, wu, wd))


def test_matches_reference_formula():
    torch.manual_seed(1)
    wg, wu, wd = torch.randn(12, 4), torch.randn(12, 4), torch.randn(4, 12)
    x = torch.randn(3, 4)
    ref = F.linear(F.silu(F.linear(x, wg)) * F.linear(x, wu), wd)
    torch.testing.assert_close(swiglu_ffn(x, wg, wu, wd), ref)


def test_parameter_share():
    per_layer = 3 * 896 * 4864
    assert per_layer == 13_074_432
    assert round(24 * per_layer / 494_032_768, 3) == 0.635


@pytest.fixture(scope="module")
def qwen():
    try:
        from common.qwen import load_model, load_tokenizer
        return load_model(True), load_tokenizer(True)
    except Exception:
        pytest.skip("Qwen 모델이 아직 내려받아지지 않음(swiglu.py를 먼저 실행)")


@pytest.mark.parametrize("layer", [0, 23])
def test_qwen_mlp_reproduction(qwen, layer):
    from swiglu import TEXT, ffn_weights
    model, tok = qwen
    cap = {}
    handle = model.model.layers[layer].mlp.register_forward_hook(
        lambda mod, inp, out: cap.update(x=inp[0].detach(), y=out.detach()))
    with torch.no_grad():
        model(tok(TEXT, return_tensors="pt").input_ids)
    handle.remove()
    torch.testing.assert_close(swiglu_ffn(cap["x"], *ffn_weights(model, layer)), cap["y"],
                               atol=1e-4, rtol=1e-5)
