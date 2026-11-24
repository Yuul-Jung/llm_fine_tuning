# 제19장 실습 1: 제12~18장에서 직접 만든 부품으로 Qwen2-Tiny 전체 모델을 조립한다
#
# 실행: python examples/ch19/tiny_model.py
#
# 계층 이름(model.layers.0.self_attn.q_proj …)은 Hugging Face Qwen2와 똑같이 맞춘다.
# 이름과 Shape가 같아야 제05절에서 가중치를 그대로 옮겨 실을 수 있다.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "ch05", ROOT / "ch03", ROOT / "ch14", ROOT / "ch15",
          ROOT / "ch16", ROOT / "ch17", ROOT / "ch18"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from linear_scratch import linear_forward  # noqa: E402  (제05장)
from mha_gqa import grouped_attention  # noqa: E402     (제15장)
from norm_residual import rms_norm  # noqa: E402        (제18장)
from rope import apply_rope, rope_cos_sin  # noqa: E402 (제16장)
from swiglu import swiglu_ffn  # noqa: E402             (제17장)

TINY = dict(vocab_size=151_936, hidden_size=256, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=1, intermediate_size=768,
            rope_theta=1e6, rms_norm_eps=1e-6)


class Norm(nn.Module):
    """RMSNorm의 학습 가중치(γ)만 담는 그릇. 계산은 제18장 rms_norm이 한다."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        h, d = c["hidden_size"], c["hidden_size"] // c["num_attention_heads"]
        kv = c["num_key_value_heads"] * d
        self.q_proj = nn.Linear(h, h, bias=True)       # Qwen2: Q·K·V에만 편향
        self.k_proj = nn.Linear(h, kv, bias=True)
        self.v_proj = nn.Linear(h, kv, bias=True)
        self.o_proj = nn.Linear(h, h, bias=False)


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        h, i = c["hidden_size"], c["intermediate_size"]
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn, self.mlp = Attention(c), MLP(c)
        self.input_layernorm = Norm(c["hidden_size"])
        self.post_attention_layernorm = Norm(c["hidden_size"])


class Body(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.layers = nn.ModuleList(Block(c) for _ in range(c["num_hidden_layers"]))
        self.norm = Norm(c["hidden_size"])


class Qwen2Tiny(nn.Module):
    def __init__(self, c=TINY):
        super().__init__()
        self.c = c
        self.model = Body(c)
        self.lm_head = nn.Linear(c["hidden_size"], c["vocab_size"], bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight     # Weight Tying: 같은 텐서

    def block(self, b, x, cos, sin):
        c, eps = self.c, self.c["rms_norm_eps"]
        a = b.self_attn
        n = rms_norm(x, b.input_layernorm.weight, eps)
        q = linear_forward(n, a.q_proj.weight, a.q_proj.bias)
        k = linear_forward(n, a.k_proj.weight, a.k_proj.bias)
        v = linear_forward(n, a.v_proj.weight, a.v_proj.bias)
        rope = lambda q_, k_: (apply_rope(q_, cos, sin), apply_rope(k_, cos, sin))  # noqa: E731
        o = grouped_attention(q, k, v, c["num_attention_heads"], c["num_key_value_heads"], rope=rope)
        h = x + linear_forward(o, a.o_proj.weight)                          # 잔차 1
        m = b.mlp
        n = rms_norm(h, b.post_attention_layernorm.weight, eps)
        return h + swiglu_ffn(n, m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight)  # 잔차 2

    def forward(self, ids, trace=None):
        c = self.c
        x = self.model.embed_tokens.weight[ids]                  # [B, T] → [B, T, 256]
        cos, sin = rope_cos_sin(ids.shape[1], c["hidden_size"] // c["num_attention_heads"],
                                c["rope_theta"])
        if trace is not None:
            trace.append(("Embedding", x.shape))
        for i, b in enumerate(self.model.layers):
            x = self.block(b, x, cos, sin)
            if trace is not None:
                trace.append((f"Block {i}", x.shape))
        x = rms_norm(x, self.model.norm.weight, c["rms_norm_eps"])
        logits = linear_forward(x, self.lm_head.weight)          # [B, T, 256] → [B, T, 151936]
        if trace is not None:
            trace += [("Final Norm", x.shape), ("LM Head", logits.shape)]
        return logits


def init_weights(model, seed=0, std=0.02):
    """Hugging Face 기본값처럼 N(0, 0.02)로 초기화한다. 편향도 작게 흔들어 편향 경로까지 검증한다."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():                 # 공유 텐서는 한 번만 나온다
            if name.endswith("layernorm.weight") or name == "model.norm.weight":
                p.fill_(1.0)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * std)
    return model


def count_by_hand(c=TINY):
    h, i, L, V = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["vocab_size"]
    kv = c["num_key_value_heads"] * h // c["num_attention_heads"]
    attn = (h * h + h) + 2 * (h * kv + kv) + h * h               # q(편향) + k·v(편향) + o
    mlp = 3 * h * i
    layer = attn + mlp + 2 * h                                    # 정규화 2개
    emb = V * h
    return dict(embedding=emb, attention=attn, mlp=mlp, layer=layer,
                layers=L * layer, final_norm=h, total=emb + L * layer + h)


def main() -> None:
    model = init_weights(Qwen2Tiny()).eval()

    print("── 1. 파라미터 수: 손계산과 PyTorch 집계 ──")
    hand = count_by_hand()
    unique = sum(p.numel() for p in model.parameters())          # 공유 텐서 중복 제거
    naive = sum(t.numel() for t in model.state_dict().values())  # 이름마다 세면 두 번
    for k in ("embedding", "attention", "mlp", "layer", "layers", "final_norm", "total"):
        print(f"  {k:<11}{hand[k]:>12,}")
    print(f"  PyTorch 집계(공유 1회) {unique:,} / 이름별 합산 {naive:,}")
    print(f"  Embedding 비중 {hand['embedding'] / hand['total']:.1%}, "
          f"FP32 크기 {hand['total'] * 4 / 1e6:.1f}MB")
    same = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    print(f"  lm_head와 embed_tokens가 같은 메모리: {same}")

    print("\n── 2. 차원 추적(토큰 6개) ──")
    trace = []
    with torch.no_grad():
        logits = model(torch.tensor([[9707, 11, 1879, 0, 1234, 42]]), trace=trace)
    print(f"  {'입력 ids':<11}(1, 6)")
    for name, shape in trace:
        print(f"  {name:<11}{tuple(shape)}")
    print(f"  무작위 모델의 Logits 표준편차 {logits.std().item():.3f}")


if __name__ == "__main__":
    main()
