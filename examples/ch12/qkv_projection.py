# 제12장 핵심 실습: Qwen2.5-0.5B 0번 계층의 Q·K·V Projection을 직접 계산해 모델 내부 출력과 비교
#
# 실행: python examples/ch12/qkv_projection.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ch03"))
sys.path.insert(0, str(ROOT / "ch05"))

import torch  # noqa: E402

from common.qwen import load_model, load_tokenizer  # noqa: E402
from head_split import split_heads  # noqa: E402  (제03장)
from linear_scratch import linear_forward  # noqa: E402  (제05장)

TEXT = "언어 모델은 다음 토큰을 예측한다."


def toy_retrieval() -> None:
    """검색 비유: Query와 가장 닮은 Key의 Value를 가져온다(가중 평균)."""
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])     # 색인 3개
    values = torch.tensor([[10.0], [20.0], [30.0]])                 # 각 색인의 내용
    query = torch.tensor([0.0, 5.0])                                # 두 번째 색인과 닮은 질문
    weights = torch.softmax(keys @ query, dim=0)                    # 닮은 정도 → 비율
    print(f"Query·Key 점수: {(keys @ query).tolist()}")
    print(f"가져올 비율:    {[round(w, 3) for w in weights.tolist()]}")
    print(f"가져온 내용:    {(weights @ values).item():.2f}  (Value 20에 가까움)")


def capture_projections(model, ids: torch.Tensor) -> dict:
    """Forward Hook으로 0번 계층의 RMSNorm 출력과 q/k/v_proj 출력을 붙잡는다."""
    layer = model.model.layers[0]
    captured, handles = {}, []
    targets = {"norm": layer.input_layernorm, "q": layer.self_attn.q_proj,
               "k": layer.self_attn.k_proj, "v": layer.self_attn.v_proj}
    for name, module in targets.items():
        def hook(mod, inputs, output, name=name):
            captured[name] = output.detach()
        handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()                                                   # Hook은 쓰고 나면 반드시 제거
    return captured


def main() -> None:
    print("── 1. 검색 비유로 보는 Q·K·V ──")
    toy_retrieval()

    model, tok = load_model(), load_tokenizer()
    attn = model.model.layers[0].self_attn

    print("\n── 2. 0번 계층 Q·K·V Projection 가중치 ──")
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        lin = getattr(attn, name)
        bias = tuple(lin.bias.shape) if lin.bias is not None else "없음"
        print(f"  {name}: weight {tuple(lin.weight.shape)}, bias {bias}")

    ids = tok(TEXT, return_tensors="pt").input_ids
    cap = capture_projections(model, ids)
    x = cap["norm"]                                                  # q/k/v_proj에 들어간 입력
    print(f"\n── 3. 같은 입력 {tuple(x.shape)}으로 직접 계산 ──")
    for name in ("q", "k", "v"):
        lin = getattr(attn, f"{name}_proj")
        mine = linear_forward(x, lin.weight.detach(), lin.bias.detach())
        error = (mine - cap[name]).abs().max().item()
        print(f"  {name.upper()}: 직접 {tuple(mine.shape)}, 모델 내부 {tuple(cap[name].shape)}, 최대 오차 {error:.1e}")

    print("\n── 4. 헤드로 나누기(제03장 split_heads) ──")
    q_heads = split_heads(cap["q"], 14)
    k_heads = split_heads(cap["k"], 2)
    print(f"  Q {tuple(cap['q'].shape)} → {tuple(q_heads.shape)}  (Query 헤드 14개)")
    print(f"  K {tuple(cap['k'].shape)} → {tuple(k_heads.shape)}  (KV 헤드 2개)")

    print("\n── 5. Bias의 크기 ──")
    for name in ("q", "k", "v"):
        lin = getattr(attn, f"{name}_proj")
        no_bias = (x @ lin.weight.detach().T).norm(dim=-1).mean().item()
        print(f"  {name}_proj: bias 크기 {lin.bias.norm().item():7.2f},  "
              f"bias 없는 출력의 평균 크기 {no_bias:7.2f}")


if __name__ == "__main__":
    main()
