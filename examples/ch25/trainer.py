# 제25장: 제3·4부 학습 실습이 함께 쓰는 학습 루프
#
# 한 단계 = (묶음 accum개) × [순전파 → 손실 ÷ accum → 역전파]
#           → Gradient Clipping → optimizer.step → scheduler.step → zero_grad

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch24"))

import torch  # noqa: E402

from optim_memory import warmup_cosine  # noqa: E402  (제24장)


@torch.no_grad()
def eval_loss(model, blocks: torch.Tensor, device: str) -> float:
    """블록마다 모델의 평균 손실을 구해 다시 평균(블록마다 유효 토큰 수가 같으므로 전체 평균과 같다)"""
    was_training = model.training
    model.eval()
    losses = [model(input_ids=b.to(device), labels=b.to(device)).loss.item() for b in blocks.split(1)]
    model.train(was_training)
    return sum(losses) / len(losses)


def train(model, train_blocks, eval_sets: dict, *, steps: int, batch: int = 2, accum: int = 2,
          lr: float = 1e-3, warmup: int = 10, clip: float = 1.0, weight_decay: float = 0.1,
          device: str = "cpu", seed: int = 0, log_every: int = 25) -> list:
    """학습 가능한(requires_grad=True) 파라미터만 AdamW로 학습하고 기록(list of dict)을 돌려준다."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, warmup, steps))

    def record(step, train_loss, grad_norm):
        row = {"step": step, "lr": opt.param_groups[0]["lr"], "train": train_loss, "grad_norm": grad_norm}
        for name, blocks in eval_sets.items():
            row[name] = eval_loss(model, blocks, device)
        return row

    history = [record(0, float("nan"), float("nan"))]
    for step in range(1, steps + 1):
        total = 0.0
        for _ in range(accum):                                       # Gradient Accumulation
            idx = torch.randint(0, len(train_blocks), (batch,), generator=g)
            x = train_blocks[idx].to(device)
            loss = model(input_ids=x, labels=x).loss / accum
            loss.backward()
            total += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, clip).item()   # 자르기 전 노름
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"{step}단계에서 Gradient가 NaN·Inf")
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % log_every == 0 or step == steps:
            history.append(record(step, total, grad_norm))
    model.eval()
    return history
