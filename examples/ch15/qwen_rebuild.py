# 제15장 핵심 실습: 직접 만든 부품으로 Qwen2.5-0.5B 0번 계층 Self-Attention 출력을 재현
#
# 실행: python examples/ch15/qwen_rebuild.py
#
# 사용 부품: split_heads·merge_heads(제03장), causal_attention(제14장), repeat_kv(이 장)
# 위치 회전(RoPE)은 제16장에서 직접 구현하므로, 이 장에서는 Transformers의 함수를 빌려 쓴다.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb  # noqa: E402

from common.qwen import load_model, load_tokenizer  # noqa: E402
from mha_gqa import grouped_attention  # noqa: E402

TEXT = "언어 모델은 다음 토큰을 예측한다."


def capture_layer0(model, ids) -> dict:
    """0번 계층 RMSNorm 출력(Attention 입력)과 self_attn 출력을 Hook으로 붙잡는다."""
    layer, cap = model.model.layers[0], {}
    h1 = layer.input_layernorm.register_forward_hook(
        lambda mod, inp, out: cap.__setitem__("x", out.detach()))
    h2 = layer.self_attn.register_forward_hook(
        lambda mod, inp, out: cap.__setitem__("attn", out[0].detach()))
    with torch.no_grad():
        model(ids)
    h1.remove()
    h2.remove()
    return cap


def rebuild_attention(model, x: torch.Tensor) -> torch.Tensor:
    """q·k·v_proj → 헤드 분할 → RoPE → GQA Causal Attention → 헤드 결합 → o_proj"""
    attn = model.model.layers[0].self_attn
    positions = torch.arange(x.shape[1])[None]
    cos, sin = model.model.rotary_emb(x, positions)

    def rope(q, k):
        return apply_rotary_pos_emb(q, k, cos, sin)

    with torch.no_grad():
        q, k, v = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
        merged = grouped_attention(q, k, v, n_heads=14, n_kv=2, rope=rope)
        return attn.o_proj(merged)


def main() -> None:
    model, tok = load_model(), load_tokenizer()
    ids = tok(TEXT, return_tensors="pt").input_ids
    cap = capture_layer0(model, ids)

    mine = rebuild_attention(model, cap["x"])
    print(f"입력(RMSNorm 출력) {tuple(cap['x'].shape)}")
    print(f"직접 조립한 Attention 출력 {tuple(mine.shape)}")
    print(f"모델 내부 Attention 출력   {tuple(cap['attn'].shape)}")
    print(f"최대 절대오차: {(mine - cap['attn']).abs().max().item():.1e}")


if __name__ == "__main__":
    main()
