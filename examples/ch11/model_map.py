# 제11장 핵심 실습: Qwen2.5-0.5B의 설정·계층 구조·데이터 흐름을 구조도와 대조
#
# 실행: python examples/ch11/model_map.py
# 필요: 시스템 RAM 약 3GB(FP32 모델 2GB + 여유)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.qwen import load_config, load_model, load_tokenizer  # noqa: E402

CONFIG_KEYS = ["hidden_size", "num_hidden_layers", "num_attention_heads",
               "num_key_value_heads", "intermediate_size", "vocab_size",
               "hidden_act", "rms_norm_eps", "rope_theta", "tie_word_embeddings"]


def count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def block_table(model) -> dict:
    """구조도의 블록별 파라미터 수(공유 텐서는 한 번만)."""
    layer = model.model.layers[0]
    return {
        "Embedding": count(model.model.embed_tokens),
        "Attention(계층 1개)": count(layer.self_attn),
        "FFN(계층 1개)": count(layer.mlp),
        "RMSNorm 2개(계층 1개)": count(layer.input_layernorm) + count(layer.post_attention_layernorm),
        "Final Norm": count(model.model.norm),
    }


def main() -> None:
    config = load_config()
    print("── 1. config.json 핵심 값 ──")
    for key in CONFIG_KEYS:
        value = getattr(config, key, None)
        if value is None and key == "rope_theta":
            value = config.rope_parameters["rope_theta"]
        print(f"  {key:<22} {value}")

    model = load_model()
    tok = load_tokenizer()

    print("\n── 2. 최상위 구조 ──")
    for name, child in model.named_children():
        print(f"  {name}: {type(child).__name__}")
    for name, child in model.model.named_children():
        size = len(child) if isinstance(child, torch.nn.ModuleList) else ""
        print(f"    model.{name}: {type(child).__name__} {size}")

    print("\n── 3. Decoder Layer 하나의 내부(layers.0) ──")
    for name, p in model.model.layers[0].named_parameters():
        print(f"  {name:<34} {str(tuple(p.shape)):<12}")

    print("\n── 4. 블록별 파라미터 수 ──")
    blocks = block_table(model)
    for name, n in blocks.items():
        print(f"  {name:<20} {n:>12,}")
    per_layer = blocks["Attention(계층 1개)"] + blocks["FFN(계층 1개)"] + blocks["RMSNorm 2개(계층 1개)"]
    total = blocks["Embedding"] + 24 * per_layer + blocks["Final Norm"]
    print(f"  계산 합계: {total:,}  /  모델 집계: {count(model):,}")
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    print(f"  LM Head와 Embedding이 같은 텐서: {tied}")

    print("\n── 5. 데이터 흐름(Shape 추적) ──")
    ids = tok("언어 모델은 다음 토큰을 예측한다.", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states
    print(f"  Token ID           {tuple(ids.shape)}")
    print(f"  Embedding 출력      {tuple(hs[0].shape)}")
    print(f"  Decoder Layer 출력  {tuple(hs[1].shape)} × {len(hs) - 1}개")
    print(f"  Logits             {tuple(out.logits.shape)}")
    next_id = int(out.logits[0, -1].argmax())
    print(f"  마지막 위치의 다음 토큰 예측: {tok.decode([next_id])!r}")


if __name__ == "__main__":
    main()
