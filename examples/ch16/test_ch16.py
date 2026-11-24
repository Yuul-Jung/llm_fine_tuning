# 제16장 검증 테스트
#
# 실행: python -m pytest examples/ch16 -q

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "ch15", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

from rope import apply_rope, inv_freq, rope_cos_sin, rotate_2d, score_at  # noqa: E402


def test_rotation_preserves_length():
    v = torch.tensor([3.0, 4.0])
    assert rotate_2d(v, 1.234).norm().item() == pytest.approx(5.0, abs=1e-5)
    cos, sin = rope_cos_sin(50, 64, 1e6)
    x = torch.randn(50, 64)
    torch.testing.assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1))


def test_rope_pair_matches_2d_rotation():
    # 쌍 (x[i], x[i+32])는 각도 p·inv_freq[i]로 회전한 2차원 벡터와 같다
    x = torch.randn(64)
    p, i = 7, 5
    cos, sin = rope_cos_sin(p + 1, 64, 1e6)
    y = apply_rope(x, cos[p], sin[p])
    angle = p * inv_freq(64, 1e6)[i].item()
    expected = rotate_2d(torch.stack([x[i], x[i + 32]]), angle)
    torch.testing.assert_close(torch.stack([y[i], y[i + 32]]), expected, atol=1e-5, rtol=1e-5)


def test_position_zero_is_identity():
    cos, sin = rope_cos_sin(1, 64, 1e6)
    x = torch.randn(64)
    torch.testing.assert_close(apply_rope(x, cos[0], sin[0]), x)


def test_relative_position_property():
    torch.manual_seed(0)
    q, k = torch.randn(64), torch.randn(64)
    assert score_at(q, k, 3, 1) == pytest.approx(score_at(q, k, 100, 98), abs=1e-3)
    assert score_at(q, k, 3, 1) != pytest.approx(score_at(q, k, 10, 1), abs=1e-3)


def test_matches_transformers():
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    torch.manual_seed(1)
    cos, sin = rope_cos_sin(16, 64, 1e6)
    q, k = torch.randn(1, 14, 16, 64), torch.randn(1, 2, 16, 64)
    qh, kh = apply_rotary_pos_emb(q, k, cos[None], sin[None])
    torch.testing.assert_close(apply_rope(q, cos, sin), qh)
    torch.testing.assert_close(apply_rope(k, cos, sin), kh)


def test_first_pair_period():
    assert 2 * math.pi / inv_freq(64, 1e6)[0].item() == pytest.approx(2 * math.pi)


@pytest.fixture(scope="module")
def qwen():
    try:
        from common.qwen import load_config, load_model, load_tokenizer
        return load_config(True), load_model(True), load_tokenizer(True)
    except Exception:
        pytest.skip("Qwen 모델이 아직 내려받아지지 않음(qwen_rope.py를 먼저 실행)")


def test_rebuild_with_own_rope(qwen):
    from qwen_rebuild import TEXT, capture_layer0
    from qwen_rope import rebuild_with_own_rope
    config, model, tok = qwen
    cap = capture_layer0(model, tok(TEXT, return_tensors="pt").input_ids)
    theta = config.rope_parameters["rope_theta"]
    torch.testing.assert_close(rebuild_with_own_rope(model, cap["x"], theta), cap["attn"],
                               atol=1e-5, rtol=1e-5)
