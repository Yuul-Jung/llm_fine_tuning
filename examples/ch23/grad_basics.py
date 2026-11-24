# 제23장 핵심 실습: 작은 Linear 분류기의 Gradient를 손계산·수치 미분·Autograd로 비교한다
#
# 실행: python examples/ch23/grad_basics.py
#
# 모델: z = W x + b (어휘 3개짜리 점수), 손실 = Cross Entropy(z, 정답) (제22장)

import torch
import torch.nn.functional as F

X = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)       # 입력 3차원
TARGET = 2                                                    # 정답 토큰
W0 = torch.tensor([[0.2, -0.1, 0.4],
                   [0.0, 0.3, -0.2],
                   [-0.3, 0.1, 0.1]], dtype=torch.float64)    # [출력 3, 입력 3]
B0 = torch.tensor([0.1, 0.0, -0.1], dtype=torch.float64)


def loss_fn(w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    z = w @ X + b
    return F.cross_entropy(z.unsqueeze(0), torch.tensor([TARGET]))


def grad_by_hand(w: torch.Tensor, b: torch.Tensor) -> tuple:
    """연쇄법칙: dL/dz = softmax(z) − 정답 원-핫,  dL/dW = (dL/dz) xᵀ,  dL/db = dL/dz"""
    p = torch.softmax(w @ X + b, dim=0)
    dz = p.clone()
    dz[TARGET] -= 1.0
    return torch.outer(dz, X), dz


def grad_numeric(w: torch.Tensor, b: torch.Tensor, h: float = 1e-6) -> tuple:
    """중앙 차분: (L(θ + h) − L(θ − h)) / 2h 를 원소마다 계산한다(FP64)."""
    gw, gb = torch.zeros_like(w), torch.zeros_like(b)
    for g, param in ((gw, w), (gb, b)):
        flat, gflat = param.view(-1), g.view(-1)
        for i in range(flat.numel()):
            old = flat[i].item()
            flat[i] = old + h
            up = loss_fn(w, b).item()
            flat[i] = old - h
            down = loss_fn(w, b).item()
            flat[i] = old
            gflat[i] = (up - down) / (2 * h)
    return gw, gb


def grad_autograd(w: torch.Tensor, b: torch.Tensor) -> tuple:
    w, b = w.clone().requires_grad_(True), b.clone().requires_grad_(True)
    loss_fn(w, b).backward()                                   # 계산 그래프를 거꾸로
    return w.grad, b.grad


def main() -> None:
    print("── 1. 연쇄법칙 한 줄 예: L = (w·x + b − t)², w=3, x=2, b=1, t=5 ──")
    w = torch.tensor(3.0, requires_grad=True)
    y = w * 2.0 + 1.0                       # 7
    loss = (y - 5.0) ** 2                   # 4
    loss.backward()
    print(f"  손계산 dL/dw = dL/dy × dy/dw = 2(y − t) × x = 2×2×2 = 8,  Autograd {w.grad.item():.1f}")

    print("\n── 2. Linear 분류기 Gradient 세 방법 비교(FP64) ──")
    hw, hb = grad_by_hand(W0, B0)
    nw, nb = grad_numeric(W0.clone(), B0.clone())
    aw, ab = grad_autograd(W0, B0)
    print(f"  손실 {loss_fn(W0, B0).item():.6f}, 확률 {[round(v, 4) for v in torch.softmax(W0 @ X + B0, 0).tolist()]}")
    print(f"  dL/db  손계산 {[round(v, 6) for v in hb.tolist()]}")
    print(f"         Autograd {[round(v, 6) for v in ab.tolist()]}")
    print(f"  dL/dW 첫 행 손계산 {[round(v, 6) for v in hw[0].tolist()]}")
    for name, (gw, gb) in (("수치 미분", (nw, nb)), ("Autograd", (aw, ab))):
        err = max((gw - hw).abs().max().item(), (gb - hb).abs().max().item())
        print(f"  손계산과의 최대 차이({name}): {err:.1e}")

    print("\n── 3. gradcheck(FP64 수치 미분 vs 자동 미분) ──")
    w = W0.clone().requires_grad_(True)
    b = B0.clone().requires_grad_(True)
    print(f"  통과: {torch.autograd.gradcheck(loss_fn, (w, b))}")

    print("\n── 4. .grad는 더해진다 ──")
    w = W0.clone().requires_grad_(True)
    loss_fn(w, B0).backward()
    once = w.grad.clone()
    loss_fn(w, B0).backward()                                  # 초기화 없이 한 번 더
    print(f"  두 번째 backward 후 / 첫 번째: {(w.grad / once)[0, 0].item():.1f}배")
    w.grad = None                                              # optimizer.zero_grad()와 같은 일
    loss_fn(w, B0).backward()
    print(f"  초기화 후 다시 계산: 첫 번째와 같음 {torch.equal(w.grad, once)}")


if __name__ == "__main__":
    main()
