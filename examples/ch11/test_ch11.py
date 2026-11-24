# 제11장 검증 테스트
#
# 실행: python -m pytest examples/ch11 -q
# Qwen 관련 테스트는 모델 파일이 이미 내려받아져 있을 때만 실행한다.

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnn_vs_attention import attention_forward, rnn_forward  # noqa: E402


def test_sequential_steps():
    x = torch.randn(32, 8)
    assert rnn_forward(x, torch.randn(8, 8), torch.randn(8, 8))[1] == 32
    assert attention_forward(x)[1] == 1


def test_attention_is_causal():
    x = torch.randn(6, 8)
    y = x.clone()
    y[-1] = torch.randn(8)
    torch.testing.assert_close(attention_forward(x)[0][:-1], attention_forward(y)[0][:-1])


@pytest.fixture(scope="module")
def qwen():
    try:
        from common.qwen import load_config, load_model, load_tokenizer
        return load_config(True), load_model(True), load_tokenizer(True)
    except Exception:
        pytest.skip("Qwen 모델이 아직 내려받아지지 않음(model_map.py를 먼저 실행)")


def test_config_values(qwen):
    config, _, _ = qwen
    assert (config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
            config.num_key_value_heads, config.intermediate_size) == (896, 24, 14, 2, 4864)
    assert config.tie_word_embeddings is True


def test_structure_names(qwen):
    _, model, _ = qwen
    names = [n for n, _ in model.model.layers[0].named_parameters()]
    expected = ["self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight",
                "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias",
                "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight"]
    assert names == expected
    assert len(model.model.layers) == 24


def test_block_sums_and_tying(qwen):
    from model_map import block_table, count
    _, model, _ = qwen
    b = block_table(model)
    per_layer = b["Attention(계층 1개)"] + b["FFN(계층 1개)"] + b["RMSNorm 2개(계층 1개)"]
    assert b["Embedding"] + 24 * per_layer + b["Final Norm"] == count(model) == 494_032_768
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_data_flow_shapes(qwen):
    _, model, tok = qwen
    ids = tok("언어 모델", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    assert len(out.hidden_states) == 25                 # Embedding 출력 + 24계층
    assert all(h.shape == (1, ids.shape[1], 896) for h in out.hidden_states)
    assert out.logits.shape == (1, ids.shape[1], 151_936)
