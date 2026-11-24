# 제26장 실습 1: requires_grad=False인 가중치는 Gradient도 Optimizer 상태도 만들지 않는다
#
# 실행: python examples/ch26/freeze_demo.py

import torch


def build():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))


def one_step(model, params) -> torch.optim.Optimizer:
    opt = torch.optim.AdamW(params, lr=0.1)
    x, y = torch.randn(5, 8), torch.randint(0, 4, (5,))
    torch.nn.functional.cross_entropy(model(x), y).backward()
    opt.step()
    return opt


def main() -> None:
    model = build()
    first, last = model[0], model[2]
    first.weight.requires_grad_(False)                 # 첫 Linear 가중치를 얼린다
    first.bias.requires_grad_(False)
    w0 = first.weight.detach().clone()

    print("── 1. 얼린 가중치에는 Gradient가 생기지 않는다 ──")
    opt = one_step(model, model.parameters())          # 일부러 전체 파라미터를 넘겨도
    print(f"  첫 Linear .grad: {first.weight.grad}, 마지막 Linear .grad 모양: {tuple(last.weight.grad.shape)}")
    print(f"  첫 Linear 가중치 그대로: {torch.equal(first.weight, w0)}")

    print("\n── 2. Optimizer 상태도 학습하는 가중치에만 생긴다 ──")
    with_state = [p for p in model.parameters() if p in opt.state]
    n_state = sum(p.numel() for p in with_state)
    print(f"  상태가 있는 텐서 {len(with_state)}개, 원소 {n_state}개 "
          f"(마지막 Linear 16×4 + 4 = {16 * 4 + 4})")

    print("\n── 3. 학습할 것만 Optimizer에 넘기는 것이 원칙 ──")
    model = build()
    for p in model[0].parameters():
        p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  학습 파라미터 {sum(p.numel() for p in trainable)} / 전체 {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()
