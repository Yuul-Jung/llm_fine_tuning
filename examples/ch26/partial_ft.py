# 제26장 핵심 실습: 사전학습한 Qwen2-Tiny를 세 가지 범위로 도메인 학습하고 비교한다
#
# 실행: python examples/ch26/partial_ft.py   (먼저 제25장 pretrain_tiny.py로 사전학습 모델을 만든다)
#
# 범위: 전체(Full FT) / 마지막 Block만 / RMSNorm만. 학습 설정은 제25장 Full FT와 같다.

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "ch24", HERE.parent / "ch25"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402
from transformers import Qwen2ForCausalLM  # noqa: E402
from transformers.utils import logging  # noqa: E402

from common.device import select_device  # noqa: E402
from common.fixed_data import load_blocks  # noqa: E402
from common.qwen import load_tokenizer  # noqa: E402
from trainer import eval_loss, train  # noqa: E402     (제25장)

logging.disable_progress_bar()
PRETRAINED = HERE.parents[1] / "models" / "qwen2-tiny-pretrained"

SCOPES = {
    "전체(Full FT)": lambda name: True,
    "마지막 Block": lambda name: name.startswith("model.layers.3."),
    "RMSNorm만": lambda name: "norm" in name,
}


def freeze_except(model, keep) -> tuple:
    """keep(name)이 참인 파라미터만 학습하고 나머지는 얼린다. (학습 수, 전체 수)를 돌려준다."""
    for name, p in model.named_parameters():
        p.requires_grad_(keep(name))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def snapshot(model) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def main() -> None:
    device, _ = select_device()
    tok = load_tokenizer()
    data = {s: load_blocks(tok, s) for s in ("eval", "domain_train", "domain_eval")}
    base = Qwen2ForCausalLM.from_pretrained(PRETRAINED, dtype=torch.float32)
    ppl0 = {k: math.exp(eval_loss(base, data[k], device)) for k in ("domain_eval", "eval")}
    print(f"사전학습 모델: 도메인 PPL {ppl0['domain_eval']:.1f}, 일반 PPL {ppl0['eval']:.1f}\n")

    print(f"── 범위별 학습(60단계, lr 3e-4, 제25장 Full FT와 같은 설정) ──")
    rows = []
    for label, keep in SCOPES.items():
        model = Qwen2ForCausalLM.from_pretrained(PRETRAINED, dtype=torch.float32)
        trainable, total = freeze_except(model, keep)
        before = snapshot(model)
        train(model, data["domain_train"], {}, steps=60, lr=3e-4, warmup=5, device=device, log_every=60)
        after = snapshot(model)
        changed = [n for n in before if not torch.equal(before[n], after[n].cpu())]
        frozen_ok = all(keep(n) for n in changed)                  # 바뀐 것은 모두 학습 대상인가
        ppl = {k: math.exp(eval_loss(model, data[k], device)) for k in ("domain_eval", "eval")}
        extra = trainable * 12                                       # Gradient 4 + AdamW 8바이트
        rows.append((label, trainable, total, len(changed), frozen_ok, extra, ppl))

    for label, trainable, total, n_changed, ok, extra, ppl in rows:
        print(f"\n  [{label}]")
        print(f"    학습 파라미터 {trainable:,} / {total:,} ({trainable / total:.2%}), 바뀐 텐서 {n_changed}개")
        print(f"    동결 가중치 전부 그대로: {ok}")
        print(f"    Gradient + AdamW 상태 {extra / 2**20:,.2f}MiB")
        print(f"    도메인 PPL {ppl0['domain_eval']:.1f} → {ppl['domain_eval']:.1f}, "
              f"일반 PPL {ppl0['eval']:.1f} → {ppl['eval']:.1f}")


if __name__ == "__main__":
    main()
