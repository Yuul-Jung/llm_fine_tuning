# 제17장 핵심 실습: SwiGLU FFN을 직접 구현해 Qwen2.5 0번 계층 MLP 출력과 비교
#
# 실행: python examples/ch17/swiglu.py
#
# FFN(x) = down( silu(gate(x)) ⊙ up(x) )     ⊙: 원소별 곱(제04장)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "ch05", ROOT / "ch06"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from activations import silu  # noqa: E402  (제06장)
from common.qwen import load_model, load_tokenizer  # noqa: E402
from linear_scratch import linear_forward  # noqa: E402  (제05장)

TEXT = "언어 모델은 다음 토큰을 예측한다."


def swiglu_ffn(x, w_gate, w_up, w_down):
    """x: [..., hidden] → [..., hidden]. 세 Linear 모두 편향 없음."""
    gate = linear_forward(x, w_gate)              # [..., intermediate]
    up = linear_forward(x, w_up)                  # [..., intermediate]
    return linear_forward(silu(gate) * up, w_down)  # 원소별 곱 → 다시 hidden으로


def capture_mlp(model, ids) -> dict:
    """0번 계층 mlp의 입력과 출력을 Hook으로 붙잡는다."""
    cap = {}
    handle = model.model.layers[0].mlp.register_forward_hook(
        lambda mod, inp, out: cap.update(x=inp[0].detach(), y=out.detach()))
    with torch.no_grad():
        model(ids)
    handle.remove()
    return cap


def ffn_weights(model, layer: int = 0) -> tuple:
    mlp = model.model.layers[layer].mlp
    return tuple(getattr(mlp, n).weight.detach() for n in ("gate_proj", "up_proj", "down_proj"))


def main() -> None:
    model, tok = load_model(), load_tokenizer()
    w_gate, w_up, w_down = ffn_weights(model)

    print("── 1. 세 Projection의 모양 ──")
    for name, w in (("gate_proj", w_gate), ("up_proj", w_up), ("down_proj", w_down)):
        print(f"  {name:<10} {tuple(w.shape)}")

    cap = capture_mlp(model, tok(TEXT, return_tensors="pt").input_ids)
    print("\n── 2. 직접 구현과 모델 내부 출력 비교 ──")
    mine = swiglu_ffn(cap["x"], w_gate, w_up, w_down)
    print(f"  입력 {tuple(cap['x'].shape)} → 출력 {tuple(mine.shape)}")
    print(f"  최대 절대오차: {(mine - cap['y']).abs().max().item():.1e}")

    print("\n── 3. FFN은 토큰마다 따로 계산된다 ──")
    one_by_one = torch.cat([swiglu_ffn(cap["x"][:, t:t + 1], w_gate, w_up, w_down)
                            for t in range(cap["x"].shape[1])], dim=1)
    print(f"  토큰을 하나씩 넣은 결과 = 한꺼번에 넣은 결과: {torch.allclose(one_by_one, mine, atol=1e-5)}")

    print("\n── 4. 게이트의 효과: SiLU(gate) 값의 분포 ──")
    g = torch.nn.functional.silu(linear_forward(cap["x"], w_gate))
    near_zero = (g.abs() < 0.05).float().mean().item()
    print(f"  |SiLU(gate)| < 0.05 인 칸의 비율: {near_zero:.1%}  (이 칸들의 up 값은 거의 통과하지 못함)")

    print("\n── 5. 파라미터와 연산량 ──")
    per_layer = sum(w.numel() for w in (w_gate, w_up, w_down))
    total = 494_032_768
    print(f"  계층 1개 FFN 파라미터: {per_layer:,}  (3 × 896 × 4,864)")
    print(f"  24계층 FFN 합계: {24 * per_layer:,} = 전체의 {24 * per_layer / total:.1%}")
    print(f"  토큰 1개당 FFN 연산(24계층): 약 {2 * 24 * per_layer / 1e9:.2f} GFLOP")


if __name__ == "__main__":
    main()
