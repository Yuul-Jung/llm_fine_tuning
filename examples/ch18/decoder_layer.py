# 제18장 핵심 실습: 직접 만든 부품만으로 Qwen2.5 Decoder Layer를 조립해 실제 출력과 비교
#
# 실행: python examples/ch18/decoder_layer.py
#
# h   = x + Attention( RMSNorm₁(x) )        잔차 연결 1
# out = h + FFN( RMSNorm₂(h) )              잔차 연결 2

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "ch05", ROOT / "ch06", ROOT / "ch03", ROOT / "ch14", ROOT / "ch15",
          ROOT / "ch16", ROOT / "ch17", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from common.qwen import load_model, load_tokenizer  # noqa: E402
from linear_scratch import linear_forward  # noqa: E402  (제05장)
from mha_gqa import grouped_attention  # noqa: E402     (제15장)
from norm_residual import rms_norm  # noqa: E402        (이 장)
from rope import apply_rope, rope_cos_sin  # noqa: E402 (제16장)
from swiglu import swiglu_ffn  # noqa: E402             (제17장)

TEXT = "언어 모델은 다음 토큰을 예측한다."


def get(module, name):
    return getattr(module, name).detach()


def self_attention(attn, x, cos, sin):
    """q·k·v_proj(편향 포함) → RoPE·GQA Causal Attention → o_proj(편향 없음)"""
    q = linear_forward(x, get(attn.q_proj, "weight"), get(attn.q_proj, "bias"))
    k = linear_forward(x, get(attn.k_proj, "weight"), get(attn.k_proj, "bias"))
    v = linear_forward(x, get(attn.v_proj, "weight"), get(attn.v_proj, "bias"))
    rope = lambda q_, k_: (apply_rope(q_, cos, sin), apply_rope(k_, cos, sin))  # noqa: E731
    return linear_forward(grouped_attention(q, k, v, 14, 2, rope=rope), get(attn.o_proj, "weight"))


def decoder_layer(layer, x, cos, sin, residual: bool = True):
    """Pre-Norm Decoder Layer 하나. residual=False면 잔차 연결을 뺀 비교용 계산."""
    a = self_attention(layer.self_attn, rms_norm(x, get(layer.input_layernorm, "weight")), cos, sin)
    h = x + a if residual else a
    mlp = layer.mlp
    f = swiglu_ffn(rms_norm(h, get(layer.post_attention_layernorm, "weight")),
                   get(mlp.gate_proj, "weight"), get(mlp.up_proj, "weight"), get(mlp.down_proj, "weight"))
    return h + f if residual else f


def hidden_states(model, tok):
    ids = tok(TEXT, return_tensors="pt").input_ids
    with torch.no_grad():
        return model(ids, output_hidden_states=True).hidden_states


def main() -> None:
    model, tok = load_model(), load_tokenizer()
    hs = hidden_states(model, tok)                 # hs[i] = i번 계층의 입력
    seq = hs[0].shape[1]
    cos, sin = rope_cos_sin(seq, 64, 1e6)

    print("── 1. Decoder Layer 재현(계층 입력 hs[i] → 계층 출력 hs[i+1]) ──")
    with torch.no_grad():
        for i in (0, 12, 22):
            mine = decoder_layer(model.model.layers[i], hs[i], cos, sin)
            print(f"  {i:>2}번 계층: 최대 오차 {(mine - hs[i + 1]).abs().max().item():.1e}, "
                  f"상대 오차 {((mine - hs[i + 1]).norm() / hs[i + 1].norm()).item():.1e}")

    print("\n── 2. 마지막 계층 + Final Norm ──")
    with torch.no_grad():
        last = decoder_layer(model.model.layers[23], hs[23], cos, sin)
        final = rms_norm(last, model.model.norm.weight.detach())
    print(f"  Final Norm 결과와 모델의 마지막 hidden state 최대 오차: {(final - hs[24]).abs().max().item():.1e}")

    print("\n── 3. 잔차 연결을 빼면? ──")
    with torch.no_grad():
        no_res = decoder_layer(model.model.layers[0], hs[0], cos, sin, residual=False)
    print(f"  0번 계층 출력 상대 오차: {((no_res - hs[1]).norm() / hs[1].norm()).item():.2f}")


if __name__ == "__main__":
    main()
