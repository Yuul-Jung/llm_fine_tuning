# 제22장 실습 1: Cross Entropy를 손으로 계산하고, 한 칸 이동과 −100 마스킹을 확인한다
#
# 실행: python examples/ch22/loss_basics.py

import math

import torch
import torch.nn.functional as F

IGNORE = -100                                   # PyTorch·Hugging Face가 약속한 "계산 제외" 표시


def nll_by_hand(logits: list, target: int) -> float:
    """정답 토큰 확률의 −log. Softmax(제06장)를 직접 계산한다."""
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    return -math.log(exps[target] / sum(exps))


def shift(ids: torch.Tensor) -> tuple:
    """위치 t의 출력이 t+1번 토큰을 맞히도록 입력과 정답을 한 칸 어긋나게 맞춘다."""
    return ids[..., :-1], ids[..., 1:]


def lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple:
    """logits [B, T, V], labels [B, T](같은 위치끼리 이미 정렬됨) → (NLL 합, 유효 토큰 수)"""
    nll_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1),
                              ignore_index=IGNORE, reduction="sum")
    return nll_sum, int((labels != IGNORE).sum())


def main() -> None:
    print("── 1. 어휘 4개의 점수 [2.0, 1.0, 3.0, 0.5]에서 정답별 손실 ──")
    logits = [2.0, 1.0, 3.0, 0.5]
    for target in (2, 0, 3):
        p = math.exp(-nll_by_hand(logits, target))
        print(f"  정답 {target}: 확률 {p:.4f} → 손실 −log p = {nll_by_hand(logits, target):.4f}")
    torch_loss = F.cross_entropy(torch.tensor([logits]), torch.tensor([2])).item()
    print(f"  PyTorch cross_entropy(정답 2) = {torch_loss:.4f}")

    print("\n── 2. 한 칸 이동 ──")
    ids = torch.tensor([[11, 22, 33, 44, 55]])
    inputs, labels = shift(ids)
    for t in range(inputs.shape[1]):
        print(f"  위치 {t}: 지금까지 {ids[0, :t + 1].tolist()} → 맞힐 정답 {labels[0, t].item()}")

    print("\n── 3. −100 마스킹 ──")
    torch.manual_seed(0)
    lg = torch.randn(1, 4, 6)
    lab = torch.tensor([[1, 4, 2, 5]])
    masked = lab.clone()
    masked[0, 2:] = IGNORE                                # 뒤 두 위치는 PAD라고 가정
    s_all, n_all = lm_loss(lg, lab)
    s_m, n_m = lm_loss(lg, masked)
    per = F.cross_entropy(lg[0], lab[0], reduction="none")
    print(f"  위치별 손실 {[round(v, 4) for v in per.tolist()]}")
    print(f"  전체 평균 {s_all.item() / n_all:.4f} (토큰 {n_all}개)")
    print(f"  마스킹 평균 {s_m.item() / n_m:.4f} (토큰 {n_m}개) = 앞 두 위치 평균 {per[:2].mean().item():.4f}")


if __name__ == "__main__":
    main()
