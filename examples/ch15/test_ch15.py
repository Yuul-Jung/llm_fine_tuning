# 제15장 검증 테스트
#
# 실행: python -m pytest examples/ch15 -q

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "ch03", ROOT / "ch14", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

from head_split import merge_heads, split_heads  # noqa: E402
from mha_gqa import grouped_attention, kv_numbers, repeat_kv  # noqa: E402


def test_repeat_kv_order():
    kv = torch.arange(2).float().view(1, 2, 1, 1)
    assert repeat_kv(kv, 7)[0, :, 0, 0].tolist() == [0.0] * 7 + [1.0] * 7


@pytest.mark.parametrize("n_kv", [4, 2, 1])
def test_grouped_matches_pytorch(n_kv):
    torch.manual_seed(n_kv)
    n_heads, hidden = 4, 16
    q = torch.randn(2, 5, hidden)
    k = torch.randn(2, 5, n_kv * hidden // n_heads)
    v = torch.randn(2, 5, n_kv * hidden // n_heads)
    ref = F.scaled_dot_product_attention(split_heads(q, n_heads), split_heads(k, n_kv),
                                         split_heads(v, n_kv), is_causal=True, enable_gqa=True)
    torch.testing.assert_close(grouped_attention(q, k, v, n_heads, n_kv), merge_heads(ref))


def test_gqa_with_repeated_kv_equals_mha():
    # KV 헤드를 미리 복제해 MHA로 계산해도 같은 결과
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 6, 16), torch.randn(1, 6, 8), torch.randn(1, 6, 8)
    k_full = merge_heads(repeat_kv(split_heads(k, 2), 2))
    v_full = merge_heads(repeat_kv(split_heads(v, 2), 2))
    torch.testing.assert_close(grouped_attention(q, k, v, 4, 2),
                               grouped_attention(q, k_full, v_full, 4, 4))


def test_qwen_kv_numbers():
    params, cache = kv_numbers(896, 14, 2)
    assert params == 2 * 896 * 128
    assert cache == 512                          # 2 × 2 × 64 × 2바이트
    assert cache * 24 == 12 * 1024               # 24계층 → 토큰당 12KiB
    assert kv_numbers(896, 14, 14)[1] == 7 * cache


@pytest.fixture(scope="module")
def qwen():
    try:
        from common.qwen import load_model, load_tokenizer
        return load_model(True), load_tokenizer(True)
    except Exception:
        pytest.skip("Qwen 모델이 아직 내려받아지지 않음(qwen_rebuild.py를 먼저 실행)")


def test_rebuild_qwen_layer0_attention(qwen):
    from qwen_rebuild import TEXT, capture_layer0, rebuild_attention
    model, tok = qwen
    cap = capture_layer0(model, tok(TEXT, return_tensors="pt").input_ids)
    torch.testing.assert_close(rebuild_attention(model, cap["x"]), cap["attn"],
                               atol=1e-5, rtol=1e-5)
