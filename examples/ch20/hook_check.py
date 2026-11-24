# 제20장 핵심 실습 1: Forward Hook으로 0번 계층 Attention·FFN 출력을 꺼내 직접 구현과 비교
#
# 실행: python examples/ch20/hook_check.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "ch05", ROOT / "ch03", ROOT / "ch14", ROOT / "ch15",
          ROOT / "ch16", ROOT / "ch17", ROOT / "ch18"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from common.qwen import load_model, load_tokenizer  # noqa: E402
from decoder_layer import get, self_attention  # noqa: E402  (제18장)
from norm_residual import rms_norm  # noqa: E402            (제18장)
from rope import rope_cos_sin  # noqa: E402                 (제16장)
from swiglu import swiglu_ffn  # noqa: E402                 (제17장)

TEXT = "훅은 모델 안을 들여다보는 창문이다."


def capture(model, ids, names):
    """이름으로 고른 모듈들에 Hook을 걸어 입력과 출력을 저장하고, 끝나면 Hook을 뗀다."""
    store, handles = {}, []
    modules = dict(model.named_modules())
    for name in names:
        def hook(module, args, output, name=name):
            out = output[0] if isinstance(output, tuple) else output
            store[name] = (args[0].detach() if args else None, out.detach())
        handles.append(modules[name].register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in handles:
            h.remove()                                    # 떼지 않으면 다음 실행에도 계속 걸린다
    return store


def main() -> None:
    model, tok = load_model(), load_tokenizer()
    ids = tok(TEXT, return_tensors="pt").input_ids
    layer = model.model.layers[0]
    names = ["model.layers.0.input_layernorm", "model.layers.0.self_attn",
             "model.layers.0.post_attention_layernorm", "model.layers.0.mlp"]
    got = capture(model, ids, names)

    print("── 1. Hook으로 꺼낸 값 ──")
    for n in names:
        print(f"  {n:<40}출력 {tuple(got[n][1].shape)}")
    print(f"  Hook 해제 후 남은 Hook 수: {len(layer.self_attn._forward_hooks)}")

    cos, sin = rope_cos_sin(ids.shape[1], 64, 1e6)
    with torch.no_grad():
        x = model.model.embed_tokens(ids)                 # 0번 계층의 입력
        n1 = rms_norm(x, get(layer.input_layernorm, "weight"))
        attn = self_attention(layer.self_attn, n1, cos, sin)
        h = x + attn
        n2 = rms_norm(h, get(layer.post_attention_layernorm, "weight"))
        m = layer.mlp
        ffn = swiglu_ffn(n2, get(m.gate_proj, "weight"), get(m.up_proj, "weight"),
                         get(m.down_proj, "weight"))

    print("\n── 2. 직접 구현(제13~18장)과 비교 ──")
    for label, mine, n in (("RMSNorm 1", n1, names[0]), ("Attention", attn, names[1]),
                           ("RMSNorm 2", n2, names[2]), ("FFN", ffn, names[3])):
        ref = got[n][1]
        print(f"  {label:<10}최대 오차 {(mine - ref).abs().max().item():.1e}, "
              f"상대 오차 {((mine - ref).norm() / ref.norm()).item():.1e}")


if __name__ == "__main__":
    main()
