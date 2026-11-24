# 제24장 핵심 실습: 같은 Bigram 학습 문제를 SGD·Adam·AdamW로 풀어 손실 곡선을 비교한다
#
# 실행: python examples/ch24/optim_compare.py
#
# 문제: 고정 학습 데이터(제21장)에서 "앞 토큰 → 다음 토큰" 점수표를 경사하강법으로 학습한다.
# 제07장의 Bigram을 세는 대신 Gradient로 배운다. 어휘는 데이터에 나온 토큰만 남겨 줄인다.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common.device import select_device  # noqa: E402
from common.fixed_data import load_blocks  # noqa: E402
from common.qwen import load_tokenizer  # noqa: E402

STEPS, BATCH, SEED = 300, 4, 0
CHECKPOINTS = (0, 10, 30, 100, 300)


def compact(train: torch.Tensor, evals: torch.Tensor) -> tuple:
    """학습 데이터에 나온 토큰만 0, 1, 2, …로 번호를 다시 매긴다. 처음 보는 평가 토큰은 −1."""
    vocab = train.unique()
    lookup = torch.full((151_936,), -1, dtype=torch.long)
    lookup[vocab] = torch.arange(len(vocab))
    return lookup[train], lookup[evals], len(vocab)


def eval_loss(table, evals: torch.Tensor, v: int) -> float:
    """처음 보는 토큰이 낀 위치는 −100으로 빼고(제22장) 평균 손실을 잰다."""
    x, y = evals[:, :-1], evals[:, 1:].clone()
    y[(x < 0) | (y < 0)] = -100
    with torch.no_grad():
        return F.cross_entropy(table(x.clamp(min=0)).reshape(-1, v), y.reshape(-1),
                               ignore_index=-100).item()


def train(blocks: torch.Tensor, evals: torch.Tensor, v: int, make_opt, device: str) -> tuple:
    """점수표 [v, v]를 학습하고 (기록된 학습 손실, 평가 손실, 최종 가중치 노름)을 돌려준다."""
    torch.manual_seed(SEED)
    table = torch.nn.Embedding(v, v).to(device)            # 행 = 앞 토큰, 열 = 다음 토큰의 점수
    opt = make_opt(table.parameters())
    g = torch.Generator().manual_seed(SEED)                 # 모든 Optimizer가 같은 묶음 순서
    blocks, evals = blocks.to(device), evals.to(device)
    log = {}
    for step in range(STEPS + 1):
        idx = torch.randint(0, len(blocks), (BATCH,), generator=g)
        x = blocks[idx]
        loss = F.cross_entropy(table(x[:, :-1]).reshape(-1, v), x[:, 1:].reshape(-1))
        if step in CHECKPOINTS:
            log[step] = loss.item()
        if step == STEPS:
            break
        opt.zero_grad()                                      # 제23장: .grad 초기화
        loss.backward()
        opt.step()                                           # 가중치 갱신
    return log, eval_loss(table, evals, v), table.weight.norm().item()


def main() -> None:
    device, _ = select_device()
    tok = load_tokenizer()
    blocks, evals, v = compact(load_blocks(tok, "train"), load_blocks(tok, "eval"))
    print(f"학습 블록 {tuple(blocks.shape)}, 줄인 어휘 {v}개, 점수표 파라미터 {v * v:,}개, "
          f"묶음 {BATCH}, {STEPS}단계\n")

    runs = {
        "SGD lr=0.1": lambda p: torch.optim.SGD(p, lr=0.1),
        "SGD lr=10": lambda p: torch.optim.SGD(p, lr=10),
        "SGD+Momentum lr=10": lambda p: torch.optim.SGD(p, lr=10, momentum=0.9),
        "Adam lr=0.1": lambda p: torch.optim.Adam(p, lr=0.1),
        "AdamW lr=0.1 wd=0.1": lambda p: torch.optim.AdamW(p, lr=0.1, weight_decay=0.1),
    }
    print(f"── 1. 단계별 손실(처음 값 ln {v} = {torch.log(torch.tensor(float(v))).item():.2f} 근처) ──")
    print("  " + f"{'Optimizer':<22}" + "".join(f"{s:>7}" for s in CHECKPOINTS) + "   평가   노름")
    for name, make in runs.items():
        log, ev, norm = train(blocks, evals, v, make, device)
        print("  " + f"{name:<22}" + "".join(f"{log[s]:>7.2f}" for s in CHECKPOINTS)
              + f"{ev:>7.2f}{norm:>7.0f}")
    print("  (0~300: 학습 묶음 손실, 평가: 학습 후 평가 데이터 손실, 노름: 점수표 가중치 L2 노름)")


if __name__ == "__main__":
    main()
