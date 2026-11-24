# 제18장 검증 테스트
#
# 실행: python -m pytest examples/ch18 -q

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "ch05", ROOT / "ch06", ROOT / "ch03", ROOT / "ch14", ROOT / "ch15",
          ROOT / "ch16", ROOT / "ch17", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

from norm_residual import layer_norm, rms_norm  # noqa: E402


def test_rms_norm_hand_example():
    # [3, 4]: RMS = √((9+16)/2) = √12.5 → [3, 4] / √12.5
    x = torch.tensor([3.0, 4.0])
    torch.testing.assert_close(rms_norm(x, torch.ones(2), eps=0.0), x / 12.5 ** 0.5)


def test_norms_match_pytorch():
    torch.manual_seed(0)
    x, w, b = torch.randn(3, 5, 16) * 10, torch.randn(16), torch.randn(16)
    torch.testing.assert_close(layer_norm(x, w, b), F.layer_norm(x, (16,), w, b))
    torch.testing.assert_close(rms_norm(x, w), F.rms_norm(x, (16,), w, eps=1e-6))


def test_rms_is_one_and_scale_invariant():
    x = torch.randn(4, 32) * 7 + 2
    y = rms_norm(x, torch.ones(32))
    torch.testing.assert_close(y.pow(2).mean(dim=-1).sqrt(), torch.ones(4), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rms_norm(x * 100, torch.ones(32)), y, atol=1e-5, rtol=1e-5)


def test_layer_norm_removes_mean_rms_norm_does_not():
    x = torch.randn(4, 32) + 5
    assert layer_norm(x, torch.ones(32), torch.zeros(32)).mean(dim=-1).abs().max() < 1e-5
    assert rms_norm(x, torch.ones(32)).mean(dim=-1).abs().min() > 0.1


@pytest.fixture(scope="module")
def qwen_states():
    try:
        from common.qwen import load_model, load_tokenizer
        model, tok = load_model(True), load_tokenizer(True)
    except Exception:
        pytest.skip("Qwen 모델이 아직 내려받아지지 않음(decoder_layer.py를 먼저 실행)")
    from decoder_layer import hidden_states
    return model, hidden_states(model, tok)


def test_qwen_rmsnorm_module(qwen_states):
    model, hs = qwen_states
    norm = model.model.layers[5].input_layernorm
    with torch.no_grad():
        torch.testing.assert_close(rms_norm(hs[5], norm.weight), norm(hs[5]))


@pytest.mark.parametrize("i", [0, 12, 22])
def test_decoder_layer_reproduction(qwen_states, i):
    from decoder_layer import decoder_layer
    from rope import rope_cos_sin
    model, hs = qwen_states
    cos, sin = rope_cos_sin(hs[0].shape[1], 64, 1e6)
    with torch.no_grad():
        mine = decoder_layer(model.model.layers[i], hs[i], cos, sin)
    assert ((mine - hs[i + 1]).norm() / hs[i + 1].norm()).item() < 1e-5


def test_final_norm(qwen_states):
    from decoder_layer import decoder_layer
    from rope import rope_cos_sin
    model, hs = qwen_states
    cos, sin = rope_cos_sin(hs[0].shape[1], 64, 1e6)
    with torch.no_grad():
        final = rms_norm(decoder_layer(model.model.layers[23], hs[23], cos, sin), model.model.norm.weight)
    assert ((final - hs[24]).norm() / hs[24].norm()).item() < 1e-5
