# 제22장 핵심 실습: 직접 계산한 손실과 모델의 loss 출력을 맞추고, 고정 평가 데이터의 기준 PPL을 저장한다
#
# 실행: python examples/ch22/qwen_ppl.py
#
# 0.5B 모델(FP32 약 2GB)은 GT 1030의 VRAM에 들어가지 않으므로 두 경로 모두 CPU에서 계산한다.
# 작은 Qwen2-Tiny는 select_device()가 고른 장치에서 계산한다.

import json
import math
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch19", HERE.parent / "ch21"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402
import transformers  # noqa: E402

from base_vs_instruct import BASE_REVISION, load_base  # noqa: E402  (제21장)
from common.device import select_device  # noqa: E402
from common.fixed_data import SEQ_LEN, fingerprint, load_blocks  # noqa: E402
from common.models import QWEN_REVISION  # noqa: E402
from common.qwen import load_model, load_tokenizer  # noqa: E402
from hf_compat import to_hf  # noqa: E402               (제19장)
from loss_basics import lm_loss, shift  # noqa: E402
from tiny_model import Qwen2Tiny, init_weights  # noqa: E402

OUT = HERE / "baseline_ppl.json"


@torch.no_grad()
def evaluate(model, blocks: torch.Tensor, device: str = "cpu") -> dict:
    """블록마다 독립된 문맥(겹침 없음)으로 NLL을 모아 PPL을 계산한다."""
    model = model.to(device)
    total, count, hf_losses = 0.0, 0, []
    for block in blocks.split(1):                    # 블록 하나씩(메모리 절약)
        block = block.to(device)
        out = model(input_ids=block, labels=block)   # 모델 안에서 한 칸 이동
        inputs, labels = shift(block)
        s, n = lm_loss(out.logits[:, :-1], labels)   # 직접 계산
        total, count = total + s.item(), count + n
        hf_losses.append((out.loss.item(), s.item() / n))
    return {"nll_sum": total, "tokens": count, "loss": total / count,
            "ppl": math.exp(total / count), "hf_vs_mine": hf_losses}


def main() -> None:
    tok = load_tokenizer()
    blocks = load_blocks(tok, "eval")
    print(f"평가 데이터: 블록 {tuple(blocks.shape)}, 지문 {fingerprint(blocks)}, "
          f"예측 위치 {blocks.shape[0]} × {SEQ_LEN - 1} = {blocks.shape[0] * (SEQ_LEN - 1)}개\n")

    device, _ = select_device()
    runs = [("Qwen2-Tiny(무작위)", lambda: to_hf(init_weights(Qwen2Tiny(), seed=0)), device),
            ("Qwen2.5-0.5B Base", load_base, "cpu"),
            ("Qwen2.5-0.5B-Instruct", load_model, "cpu")]
    results = {}
    print("── 1. 모델 loss 출력 vs 직접 계산(블록 0) ──")
    for name, loader, dev in runs:
        r = evaluate(loader(), blocks, dev)
        hf, mine = r["hf_vs_mine"][0]
        print(f"  {name:<22}model.loss {hf:.6f} / 직접 {mine:.6f}  ({dev})")
        results[name] = r

    print("\n── 2. 고정 평가 데이터의 Perplexity ──")
    print(f"  {'모델':<22}{'평균 NLL':>9}{'PPL':>12}")
    for name, r in results.items():
        print(f"  {name:<22}{r['loss']:>9.4f}{r['ppl']:>12,.2f}")
    print(f"  참고: 어휘 151,936개를 고르게 찍으면 NLL = ln 151936 = {math.log(151936):.4f}")

    record = {
        "data": {"split": "eval", "shape": list(blocks.shape), "fingerprint": fingerprint(blocks),
                 "context": "블록마다 독립, 겹침 없음", "labels": "전체 토큰(첫 토큰 제외), 마스킹 없음"},
        "models": {"Qwen2.5-0.5B Base": BASE_REVISION, "Qwen2.5-0.5B-Instruct": QWEN_REVISION,
                   "Qwen2-Tiny(무작위)": "seed 0, 제19장 init_weights"},
        "results": {n: {"tokens": r["tokens"], "loss": round(r["loss"], 4), "ppl": round(r["ppl"], 2)}
                    for n, r in results.items()},
        "env": {"torch": torch.__version__, "transformers": transformers.__version__,
                "python": platform.python_version()},
    }
    OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n── 3. 기준값 저장: examples/ch22/{OUT.name} ──")


if __name__ == "__main__":
    main()
