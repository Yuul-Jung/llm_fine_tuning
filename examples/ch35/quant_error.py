# 제35장 핵심 실습: 역양자화한 가중치로 실제 계층과 모델 전체를 계산해 양자화 오차를 분석한다
#
# 실행: python examples/ch35/quant_error.py
#
# "가짜 양자화"(Fake Quantization): 양자화 → 즉시 역양자화한 FP32 가중치로 계산한다.
# 저장 형식만 다를 뿐, 실제 양자화 모델이 계산 직전에 복원하는 값과 같다.

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "ch22", HERE.parent / "ch33", HERE.parent / "ch34"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from common.fixed_data import load_blocks  # noqa: E402
from common.qwen import load_model, load_tokenizer  # noqa: E402
from int4_pack import dequant_int4, quant_int4  # noqa: E402        (제34장)
from int8_quant import dequant_symmetric, quant_symmetric  # noqa: E402  (제33장)
from qwen_ppl import evaluate  # noqa: E402                         (제22장)

METHODS = {
    "INT8 Per-Channel": lambda w: dequant_symmetric(*quant_symmetric(w, per_channel=True)),
    "INT4 Per-Channel": lambda w: dequant_int4(*quant_int4(w)),
    "INT4 Group 64": lambda w: dequant_int4(*quant_int4(w, 64), 64),
}
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def capture_input(model, ids, module):
    """module(nn.Linear)에 들어가는 입력을 Hook으로 꺼낸다(제20장)."""
    box = {}

    def hook(m, args, out):                        # 아무것도 돌려주지 않아야 출력이 바뀌지 않는다
        box["x"] = args[0].detach()

    h = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        h.remove()
    return box["x"]


@torch.no_grad()
def fake_quantize_model(model, method) -> int:
    """모든 Decoder Layer의 Linear 가중치 7종을 양자화 → 역양자화로 바꾼다. 바꾼 행렬 수를 돌려준다."""
    n = 0
    for name, mod in model.named_modules():
        if name.split(".")[-1] in LINEARS:
            mod.weight.copy_(method(mod.weight))
            n += 1
    return n


def main() -> None:
    tok = load_tokenizer()
    model = load_model()
    ids = load_blocks(tok, "eval")[:1]
    lin = model.model.layers[0].self_attn.q_proj
    w = lin.weight.detach()

    print("── 1. 가중치 오차: 평균(MSE)과 최악(최대 절대오차) ──")
    for name, f in METHODS.items():
        e = f(w) - w
        print(f"  {name:<18}MSE {e.pow(2).mean():.2e}, 최대 절대오차 {e.abs().max():.4f}, "
              f"상대 오차 {(e.norm() / w.norm()):.3f}")

    print("\n── 2. 이상치 하나의 영향(Per-Tensor INT8, 0번 행) ──")
    row = w[0:1].clone()
    base = (dequant_symmetric(*quant_symmetric(row)) - row).abs().mean().item()
    spiked = row.clone()
    spiked[0, 0] = row.abs().max() * 10                           # 값 하나만 10배로
    q, s = quant_symmetric(spiked)
    rest = (dequant_symmetric(q, s) - spiked)[0, 1:].abs().mean().item()
    zeros = (q[0, 1:] == 0).float().mean().item()
    print(f"  원래 행: 평균 절대오차 {base:.2e}")
    print(f"  값 하나를 10배: 나머지 895개의 평균 절대오차 {rest:.2e}({rest / base:.1f}배), "
          f"코드 0이 된 비율 {zeros:.0%}")

    print("\n── 3. 계층 출력 오차(0번 계층, 실제 입력, 토큰마다 상대 오차의 평균) ──")
    down = model.model.layers[0].mlp.down_proj
    xq, xd = capture_input(model, ids, lin), capture_input(model, ids, down)
    print(f"  {'':<18}{'q_proj(편향 포함)':>16}{'q_proj(W x만)':>15}{'down_proj':>11}{'가중치 오차':>11}")
    with torch.no_grad():
        F = torch.nn.functional
        for name, f in METHODS.items():
            def rel(a, b):
                return ((a - b).norm(dim=-1) / b.norm(dim=-1)).mean().item()
            with_bias = rel(F.linear(xq, f(w), lin.bias), F.linear(xq, w, lin.bias))
            no_bias = rel(F.linear(xq, f(w)), F.linear(xq, w))
            wd = down.weight.detach()
            d = rel(F.linear(xd, f(wd)), F.linear(xd, wd))
            print(f"  {name:<18}{with_bias:>16.4f}{no_bias:>15.4f}{d:>11.4f}"
                  f"{((f(w) - w).norm() / w.norm()):>11.3f}")

    print("\n── 4. 모델 전체(24계층 × Linear 7종)를 양자화한 PPL(고정 평가 데이터) ──")
    blocks = load_blocks(tok, "eval")
    print(f"  {'FP32(기준)':<18}PPL {math.exp(evaluate(model, blocks)['loss']):.2f}")
    for name, f in METHODS.items():
        m = load_model()
        n = fake_quantize_model(m, f)
        print(f"  {name:<18}PPL {math.exp(evaluate(m, blocks)['loss']):.2f}  (행렬 {n}개)")
        del m

    print("\n── 5. INT4 Group 64 가중치 오차의 분포(÷ Scale) ──")
    q4, s4 = quant_int4(w, 64)
    r = ((dequant_int4(q4, s4, 64) - w).reshape(-1, 64) / s4).flatten()
    hist = torch.histc(r, bins=10, min=-0.5, max=0.5)
    for i, c in enumerate(hist.tolist()):
        lo = -0.5 + i * 0.1
        print(f"  [{lo:+.1f}, {lo + 0.1:+.1f})  {'█' * round(40 * c / hist.max().item())}")


if __name__ == "__main__":
    main()
