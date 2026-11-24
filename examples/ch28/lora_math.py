# 제28장 핵심 실습: 작은 숫자 행렬로 LoRA 순전파를 손계산하고, 초기화·배율·병합을 확인한다
#
# 실행: python examples/ch28/lora_math.py
#
# y = W x + (α/r) · B (A x)        W: 동결, A [r, in]·B [out, r]: 학습
# 병합: W' = W + (α/r) · B A       → y = W' x  (추가 계산 없음)

import torch


def lora_forward(x, w, a, b, alpha: float, r: int):
    return x @ w.T + (alpha / r) * ((x @ a.T) @ b.T)


def merge(w, a, b, alpha: float, r: int):
    return w + (alpha / r) * (b @ a)


def lora_params(out_dim: int, in_dim: int, r: int) -> int:
    return r * (in_dim + out_dim)


def first_step_delta(r: int, alpha: float, lr: float) -> float:
    """B = 0에서 Adam 한 단계 학습한 뒤의 ‖ΔW‖ = ‖(α/r)·B A‖"""
    torch.manual_seed(0)
    out_dim, in_dim = 32, 32
    target = torch.randn(out_dim, in_dim) * 0.1
    x = torch.randn(256, in_dim)
    a = torch.nn.Parameter(torch.randn(r, in_dim) / in_dim ** 0.5)
    b = torch.nn.Parameter(torch.zeros(out_dim, r))
    opt = torch.optim.Adam([a, b], lr=lr)
    loss = ((alpha / r) * (x @ a.T) @ b.T - x @ target.T).pow(2).mean()
    loss.backward()
    opt.step()
    return ((alpha / r) * (b @ a)).norm().item()


def main() -> None:
    w = torch.tensor([[1.0, 0.0, 2.0],
                      [0.0, 1.0, -1.0]])                 # [out 2, in 3]
    x = torch.tensor([1.0, 2.0, 3.0])
    a = torch.tensor([[1.0, 1.0, 0.0]])                 # [r 1, in 3]
    b = torch.tensor([[0.5], [-1.0]])                   # [out 2, r 1]
    alpha, r = 2.0, 1

    print("── 1. 손계산(α = 2, r = 1, 배율 α/r = 2) ──")
    print("  원본  W x = [1·1 + 0·2 + 2·3, 0·1 + 1·2 − 1·3] = [7, −1]")
    print("  A x = [1 + 2 + 0] = [3]")
    print("  B (A x) = [0.5·3, −1·3] = [1.5, −3]")
    print("  y = [7, −1] + 2 × [1.5, −3] = [10, −7]")
    y = lora_forward(x, w, a, b, alpha, r)
    merged = merge(w, a, b, alpha, r)
    print(f"  PyTorch LoRA 순전파 {y.tolist()}")
    print(f"  병합 W' = W + 2·BA =\n{merged}")
    print(f"  병합 후 W' x {(x @ merged.T).tolist()}  같음: {torch.allclose(y, x @ merged.T)}")

    print("\n── 2. B = 0 초기화: 학습 첫 순간의 출력과 Gradient ──")
    torch.manual_seed(0)                                # A의 난수를 고정해 결과를 재현한다
    a0 = torch.randn(1, 3, requires_grad=True)
    b0 = torch.zeros(2, 1, requires_grad=True)
    y0 = lora_forward(x, w, a0, b0, alpha, r)
    print(f"  출력 {y0.tolist()} = 원본 W x {(x @ w.T).tolist()}")
    y0.pow(2).sum().backward()
    print(f"  dL/dA {a0.grad.tolist()}  (B가 0이라 A에는 신호가 가지 않는다)")
    print(f"  dL/dB {[round(v, 3) for v in b0.grad.flatten().tolist()]}  (B부터 움직이기 시작한다)")

    print("\n── 3. 배율 α/r은 학습률처럼 작용한다(r = 8, Adam 첫 단계 후 ‖ΔW‖) ──")
    for label, alpha_, lr in (("α = 16, lr 1e-3", 16, 1e-3), ("α = 32, lr 1e-3", 32, 1e-3),
                              ("α = 16, lr 2e-3", 16, 2e-3)):
        print(f"  {label:<18}배율 {alpha_ / 8:.0f}, ‖ΔW‖ {first_step_delta(8, alpha_, lr):.5f}")

    print("\n── 4. r = 8일 때 Qwen2.5-0.5B 한 계층의 LoRA 파라미터 ──")
    shapes = {"q_proj": (896, 896), "k_proj": (128, 896), "v_proj": (128, 896), "o_proj": (896, 896),
              "gate_proj": (4864, 896), "up_proj": (4864, 896), "down_proj": (896, 4864)}
    total_lora = total_orig = 0
    for name, (o, i) in shapes.items():
        n = lora_params(o, i, 8)
        total_lora, total_orig = total_lora + n, total_orig + o * i
        print(f"  {name:<10}{o:>5} × {i:<5}원본 {o * i:>10,}  LoRA {n:>7,}  ({n / (o * i):.2%})")
    print(f"  계층 합계: 원본 {total_orig:,}, LoRA {total_lora:,} ({total_lora / total_orig:.2%})")
    print(f"  24계층: LoRA {24 * total_lora:,}개 = 전체 494,032,768개의 {24 * total_lora / 494_032_768:.2%}")
    qv = 24 * (lora_params(896, 896, 8) + lora_params(128, 896, 8))
    print(f"  q_proj·v_proj만: {qv:,}개 ({qv / 494_032_768:.3%})")


if __name__ == "__main__":
    main()
