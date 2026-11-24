# 제24장 실습 2: Optimizer 상태 메모리를 계산하고 Qwen2-Tiny에서 실제 텐서 크기로 확인한다
#
# 실행: python examples/ch24/optim_memory.py

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "ch19"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from tiny_model import Qwen2Tiny, init_weights  # noqa: E402  (제19장)

MiB = 2 ** 20
STATE_BYTES = {"SGD": 0, "SGD+Momentum": 4, "AdamW": 8}     # FP32 기준 파라미터당 추가 바이트


def state_bytes(opt) -> int:
    return sum(v.numel() * v.element_size() for s in opt.state.values()
               for v in s.values() if torch.is_tensor(v) and v.dim() > 0)


def full_ft_budget(n_params: int, opt: str) -> dict:
    """FP32 Full Fine-tuning의 고정 메모리(활성값 제외): 가중치 + Gradient + Optimizer 상태"""
    return {"가중치": 4 * n_params, "Gradient": 4 * n_params, "상태": STATE_BYTES[opt] * n_params}


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    """학습률 배율: 처음 warmup단계는 0→1로 올리고, 이후 코사인으로 0까지 내린다(제25장에서 사용)."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def main() -> None:
    model = init_weights(Qwen2Tiny(), seed=0)
    n = sum(p.numel() for p in model.parameters())
    ids = torch.tensor([[9707, 11, 1879, 0, 1234, 42, 7, 7]])

    print(f"── 1. Qwen2-Tiny({n:,}개)에서 한 단계 학습 후 Optimizer 상태 측정 ──")
    for name, make in (("SGD", lambda p: torch.optim.SGD(p, lr=0.01)),
                       ("SGD+Momentum", lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9)),
                       ("AdamW", lambda p: torch.optim.AdamW(p, lr=1e-3))):
        opt = make(model.parameters())
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        got = state_bytes(opt)
        print(f"  {name:<13}측정 {got / MiB:7.1f}MiB  계산 {STATE_BYTES[name] * n / MiB:7.1f}MiB "
              f"(파라미터당 {got / n:.0f}바이트)")

    print("\n── 2. FP32 Full Fine-tuning 고정 메모리(활성값 제외) ──")
    for label, count in (("Qwen2-Tiny", n), ("Qwen2.5-0.5B", 494_032_768)):
        b = full_ft_budget(count, "AdamW")
        total = sum(b.values())
        parts = ", ".join(f"{k} {v / MiB:,.0f}" for k, v in b.items())
        print(f"  {label:<13}{parts} → 합계 {total / MiB:,.0f}MiB ({total / 2**30:.2f}GiB)")
    print("  GT 1030 VRAM 2GiB, 0.5B 모델은 Full Fine-tuning 불가 → 제26장부터 LoRA")

    print("\n── 3. Warmup + Cosine 학습률 배율(총 100단계, Warmup 10) ──")
    steps = (0, 4, 9, 10, 30, 55, 80, 99)
    print("  " + "  ".join(f"{s}:{warmup_cosine(s, 10, 100):.2f}" for s in steps))


if __name__ == "__main__":
    main()
