# 제20장 검증 테스트
#
# 실행: python -m pytest examples/ch20 -q

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "ch05", ROOT / "ch03", ROOT / "ch14", ROOT / "ch15",
          ROOT / "ch16", ROOT / "ch17", ROOT / "ch18", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

from sampling_kv import filter_top_k, filter_top_p, kv_bytes_per_token  # noqa: E402

QWEN = SimpleNamespace(hidden_size=896, num_attention_heads=14, num_key_value_heads=2,
                       num_hidden_layers=24)


def test_kv_cache_formula():
    assert kv_bytes_per_token(QWEN, 2) == 12 * 1024              # 토큰당 12KiB
    assert kv_bytes_per_token(QWEN, 2) * 32768 == 384 * 2**20     # 32K 문맥 384MiB
    # GQA가 없었다면(KV 헤드 14개) 7배
    mha = SimpleNamespace(**{**vars(QWEN), "num_key_value_heads": 14})
    assert kv_bytes_per_token(mha, 2) == 7 * kv_bytes_per_token(QWEN, 2)


def test_top_k_and_top_p_by_hand():
    p = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])
    assert filter_top_k(p, 2).tolist() == [True, True, False, False, False]
    # 누적 0.5 → 0.7 → 0.85: 0.8에 닿는 세 번째까지
    assert filter_top_p(p, 0.8).tolist() == [True, True, True, False, False]
    assert filter_top_p(p, 0.1).tolist() == [True, False, False, False, False]


def test_temperature_keeps_order_changes_sharpness():
    logits = torch.tensor([2.0, 1.0, 0.0])
    cold, hot = torch.softmax(logits / 0.5, -1), torch.softmax(logits / 2.0, -1)
    assert cold.argsort().tolist() == hot.argsort().tolist()
    assert cold[0] > hot[0]


@pytest.fixture(scope="module")
def qwen():
    try:
        from common.qwen import load_model, load_tokenizer
        return load_model(True), load_tokenizer(True)
    except Exception:
        pytest.skip("Qwen 모델이 아직 내려받아지지 않음(inspect_qwen.py를 먼저 실행)")


def test_hook_matches_own_attention(qwen):
    from decoder_layer import get, self_attention
    from hook_check import capture
    from norm_residual import rms_norm
    from rope import rope_cos_sin
    model, tok = qwen
    ids = tok("훅 검증", return_tensors="pt").input_ids
    got = capture(model, ids, ["model.layers.0.self_attn"])
    layer = model.model.layers[0]
    cos, sin = rope_cos_sin(ids.shape[1], 64, 1e6)
    with torch.no_grad():
        x = model.model.embed_tokens(ids)
        mine = self_attention(layer.self_attn, rms_norm(x, get(layer.input_layernorm, "weight")), cos, sin)
    torch.testing.assert_close(mine, got["model.layers.0.self_attn"][1], atol=1e-5, rtol=1e-4)
    assert len(layer.self_attn._forward_hooks) == 0              # Hook이 깨끗이 제거됨


def test_kv_cache_measured_and_same_output(qwen):
    from sampling_kv import greedy
    model, tok = qwen
    ids = tok("하나 둘", return_tensors="pt").input_ids
    a, cache = greedy(model, ids, 5, use_cache=True)
    b, _ = greedy(model, ids, 5, use_cache=False)
    assert a == b
    measured = sum(l.keys.nbytes + l.values.nbytes for l in cache.layers)
    assert measured == kv_bytes_per_token(QWEN, 4) * cache.get_seq_length()
