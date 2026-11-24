# 제25장 핵심 실습: Qwen2-Tiny 사전학습 → 저장 → 재로딩 → Full Fine-tuning
#
# 실행: python examples/ch25/pretrain_tiny.py
#
# 저장: models/qwen2-tiny-pretrained/, models/qwen2-tiny-bike/ (.gitignore로 커밋 제외)
# 손실 곡선: examples/ch25/loss_curve.png (.gitignore로 커밋 제외)

import json
import math
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch19", HERE.parent / "ch24"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402
from transformers import Qwen2ForCausalLM  # noqa: E402
from transformers.utils import logging  # noqa: E402

from common.device import select_device  # noqa: E402
from common.fixed_data import load_blocks  # noqa: E402
from common.qwen import load_tokenizer  # noqa: E402
from hf_compat import to_hf  # noqa: E402              (제19장)
from optim_memory import full_ft_budget  # noqa: E402  (제24장)
from tiny_model import Qwen2Tiny, init_weights  # noqa: E402
from trainer import eval_loss, train  # noqa: E402

logging.disable_progress_bar()
MODELS = HERE.parents[1] / "models"
PRE_STEPS, FT_STEPS = 300, 60


def show(history: list, keys: tuple) -> None:
    print("  " + f"{'단계':>5}{'학습률':>9}{'학습':>8}{'Grad노름':>10}" + "".join(f"{k:>9}" for k in keys))
    for r in history:
        print("  " + f"{r['step']:>5}{r['lr']:>9.1e}{r['train']:>8.3f}{r['grad_norm']:>10.2f}"
              + "".join(f"{r[k]:>9.3f}" for k in keys))


def plot(pre: list, ft: list, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["step"] for r in pre], [r["일반평가"] for r in pre], label="pretrain: eval")
    ax.plot([r["step"] for r in pre[1:]], [r["train"] for r in pre[1:]], label="pretrain: train")
    off = pre[-1]["step"]
    ax.plot([off + r["step"] for r in ft], [r["도메인평가"] for r in ft], label="full FT: domain eval")
    ax.plot([off + r["step"] for r in ft], [r["일반평가"] for r in ft], label="full FT: general eval")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    device, _ = select_device()
    tok = load_tokenizer()
    data = {s: load_blocks(tok, s) for s in ("train", "eval", "domain_train", "domain_eval")}
    print(f"장치 {device}, 블록: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in data.items()))

    model = to_hf(init_weights(Qwen2Tiny(), seed=0))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"학습 가능 파라미터 {trainable:,}개(전체), FP32·AdamW 고정 메모리 "
          f"{sum(full_ft_budget(trainable, 'AdamW').values()) / 2**20:.0f}MiB\n")

    print(f"── 1. 사전학습: Batch 2 × Accumulation 2 = 유효 4, {PRE_STEPS}단계, lr 1e-3, Warmup 20 ──")
    t0 = time.perf_counter()
    pre = train(model, data["train"], {"일반평가": data["eval"]}, steps=PRE_STEPS,
                lr=1e-3, warmup=20, device=device, log_every=50)
    show(pre, ("일반평가",))
    print(f"  걸린 시간 {time.perf_counter() - t0:.0f}초")

    print("\n── 2. Hugging Face 형식 저장 → 재로딩 → 출력 비교 ──")
    pre_dir = MODELS / "qwen2-tiny-pretrained"
    model.save_pretrained(pre_dir)
    tok.save_pretrained(pre_dir)
    reloaded = Qwen2ForCausalLM.from_pretrained(pre_dir, dtype=torch.float32).to(device).eval()
    probe = data["eval"][:1].to(device)
    with torch.no_grad():
        same = torch.equal(model(probe).logits, reloaded(probe).logits)
    print(f"  저장 파일 {sorted(f.name for f in pre_dir.iterdir())}")
    print(f"  재로딩 전후 Logits 완전히 같음: {same}")

    before = {k: eval_loss(reloaded, data[k], device) for k in ("eval", "domain_eval")}
    print(f"\n── 3. Full Fine-tuning(자전거 정비 도메인): {FT_STEPS}단계, lr 3e-4 ──")
    ft = train(reloaded, data["domain_train"], {"도메인평가": data["domain_eval"], "일반평가": data["eval"]},
               steps=FT_STEPS, lr=3e-4, warmup=5, device=device, log_every=20)
    show(ft, ("도메인평가", "일반평가"))
    after = {k: eval_loss(reloaded, data[k], device) for k in ("eval", "domain_eval")}
    print("  PPL 비교       학습 전      학습 후")
    for k, label in (("domain_eval", "도메인 평가"), ("eval", "일반 평가")):
        print(f"  {label:<10}{math.exp(before[k]):>10.1f}{math.exp(after[k]):>12.1f}")
    reloaded.save_pretrained(MODELS / "qwen2-tiny-bike")
    tok.save_pretrained(MODELS / "qwen2-tiny-bike")

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024      # Linux: KiB → MiB
    print(f"\n── 4. 메모리: 프로세스 최대 RSS {peak:,.0f}MiB(CPU, 모델 두 벌·데이터·라이브러리 포함) ──")
    if device == "cuda":
        print(f"  GPU 최대 할당 {torch.cuda.max_memory_allocated() / 2**20:,.0f}MiB")
    b = full_ft_budget(494_032_768, "AdamW")
    print(f"  0.5B Full FT 고정 메모리 {sum(b.values()) / 2**30:.2f}GiB > GT 1030 2GiB → 실행 불가")

    plot(pre, ft, HERE / "loss_curve.png")
    (HERE / "history.json").write_text(json.dumps({"pretrain": pre, "full_ft": ft}, ensure_ascii=False,
                                                  indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
