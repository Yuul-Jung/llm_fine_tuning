# 제16장 실습 2: 직접 구현한 RoPE로 바꿔 Qwen2.5 0번 계층 Attention을 다시 재현
#
# 실행: python examples/ch16/qwen_rope.py
#
# 제15장에서 빌려 쓴 apply_rotary_pos_emb 대신 이 장의 apply_rope를 쓴다.
# 이제 Attention의 모든 부품이 교재에서 직접 만든 코드다.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "ch15", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from common.qwen import load_config, load_model, load_tokenizer  # noqa: E402
from mha_gqa import grouped_attention  # noqa: E402  (제15장)
from qwen_rebuild import TEXT, capture_layer0  # noqa: E402  (제15장)
from rope import apply_rope, rope_cos_sin  # noqa: E402


def rebuild_with_own_rope(model, x: torch.Tensor, theta: float) -> torch.Tensor:
    attn = model.model.layers[0].self_attn
    cos, sin = rope_cos_sin(x.shape[1], 64, theta)            # [seq, 64]

    def rope(q, k):
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    with torch.no_grad():
        q, k, v = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
        return attn.o_proj(grouped_attention(q, k, v, 14, 2, rope=rope))


def main() -> None:
    config = load_config()
    theta = config.rope_parameters["rope_theta"]
    model, tok = load_model(), load_tokenizer()
    cap = capture_layer0(model, tok(TEXT, return_tensors="pt").input_ids)

    print(f"rope_theta = {theta:,.0f}, head_dim = {config.hidden_size // config.num_attention_heads}")
    mine = rebuild_with_own_rope(model, cap["x"], theta)
    print(f"직접 구현 RoPE로 재현한 Attention 출력과 실제 출력의 최대 오차: "
          f"{(mine - cap['attn']).abs().max().item():.1e}")

    wrong = rebuild_with_own_rope(model, cap["x"], 10_000.0)
    print(f"rope_theta를 10,000으로 잘못 쓰면 최대 오차: {(wrong - cap['attn']).abs().max().item():.2f}")


if __name__ == "__main__":
    main()
