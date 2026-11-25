# 제52장 핵심 실습: LoRA를 원본에 병합하고, 세 모델 A·B·C의 Logits와 PPL로 차이의 원인을 나눈다
#
# 실행: python examples/ch52/merge.py   (CPU 약 3분, 교재 검증 환경은 메모리 8GB)
#
# 만드는 파일: models/fpga_merged_v1/(병합한 독립 모델, 약 2GB), examples/ch52/merge.json

import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent, HERE.parent / "ch37", HERE.parent / "ch46", HERE.parent / "ch51"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.utils import logging  # noqa: E402

from common.device import select_device  # noqa: E402
from common.models import QWEN_ID, QWEN_REVISION  # noqa: E402
from load_4bit import load_nf4  # noqa: E402                     (제37장)
from save_load import FINAL, load_base, logits_of  # noqa: E402   (제51장)
from sft_loop import SFTDataset, split_rows, validate  # noqa: E402   (제46장)

logging.disable_progress_bar()
TINY = ROOT / "models" / "qwen2-tiny-pretrained"
TINY_LORA = ROOT / "models" / "qwen2-tiny-bike-lora"           # 제30장
QLORA = ROOT / "models" / "ch49_qlora"                          # 제49장
MERGED = ROOT / "models" / "fpga_merged_v1"
OUT = HERE / "merge.json"


def merge_by_hand(layer) -> torch.Tensor:
    """W' = W + (alpha / r) B A"""
    a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
    return layer.base_layer.weight + layer.scaling["default"] * (b @ a)


def diff(x: torch.Tensor, y: torch.Tensor) -> float:
    return (x - y).abs().max().item()


def ppl(model, va) -> float:
    return validate(model, va)["val_ppl"]


def main() -> None:
    device, _ = select_device()
    rec = {}

    print("── 1. 병합 수식(Qwen2-Tiny 제30장 Adapter, 0층 q_proj) ──")
    tok_t = AutoTokenizer.from_pretrained(TINY)
    tiny = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32),
                                     TINY_LORA).to(device).eval()
    layer = tiny.base_model.model.model.layers[0].self_attn.q_proj
    by_hand = merge_by_hand(layer).detach().clone()
    print(f"  r {layer.r['default']}, alpha {layer.lora_alpha['default']}, "
          f"scaling = alpha / r = {layer.scaling['default']}")
    before = logits_of(tiny, tok_t, device)

    print("\n── 2. Qwen2-Tiny 병합 전후 ──")
    merged_tiny = tiny.merge_and_unload()
    w = merged_tiny.model.layers[0].self_attn.q_proj.weight
    after = logits_of(merged_tiny, tok_t, device)
    print(f"  병합 후 모델 종류: {type(merged_tiny).__name__}(LoRA 층 없음: "
          f"{not any('lora' in n for n, _ in merged_tiny.named_modules())})")
    print(f"  merge_and_unload 가중치와 수식 계산의 최대 차이 {diff(w, by_hand):.2e}")
    print(f"  병합 전후 Logits 최대 차이 {diff(before, after):.2e}(순수 병합 오차)")
    rec["tiny"] = {"weight_diff": diff(w, by_hand), "logits_diff": diff(before, after)}
    del tiny, merged_tiny

    print("\n── 3. QLoRA Adapter(제49장): 세 모델로 차이의 원인 나누기 ──")
    tok = AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION)
    va = SFTDataset(tok, split_rows("validation"))
    b = PeftModel.from_pretrained(load_base(device), QLORA).eval()      # B: FP32 원본 + Adapter
    lb, pb = logits_of(b, tok, device), ppl(b, va)
    c = b.merge_and_unload()                                            # C: B를 병합한 독립 모델
    lc, pc = logits_of(c, tok, device), ppl(c, va)
    del b, c
    a = PeftModel.from_pretrained(load_nf4(device), QLORA).eval()      # A: 학습 때와 같은 4비트 원본 + Adapter
    la, pa = logits_of(a, tok, device), ppl(a, va)
    del a
    print("  모델                     도메인 PPL   Logits 최대 차이")
    print(f"  A 4비트 원본 + Adapter    {pa:>9.2f}   A↔B {diff(la, lb):.2f}(원본 정밀도 차이)")
    print(f"  B FP32 원본 + Adapter     {pb:>9.2f}   B↔C {diff(lb, lc):.2e}(순수 병합 오차)")
    print(f"  C B를 병합한 모델          {pc:>9.2f}")
    rec["qlora"] = {"ppl": {"A": pa, "B": pb, "C": pc}, "A_B": diff(la, lb), "B_C": diff(lb, lc)}

    print("\n── 4. CPU에서 FP32 병합(제51장 배포용 Adapter) ──")
    t0 = time.time()
    m = PeftModel.from_pretrained(load_base("cpu"), FINAL).eval()       # 원본 2GB + 병합: GT 1030 VRAM 초과
    l0, p0 = logits_of(m, tok, "cpu"), ppl(m, va)
    merged = m.merge_and_unload()
    l1, p1 = logits_of(merged, tok, "cpu"), ppl(merged, va)
    merged.save_pretrained(MERGED)
    tok.save_pretrained(MERGED)
    secs = time.time() - t0
    size = sum(f.stat().st_size for f in MERGED.glob("*.safetensors")) / 2 ** 20
    n_params = sum(p.numel() for p in merged.parameters())
    print(f"  병합 전 PPL {p0:.4f}, 병합 후 PPL {p1:.4f}, Logits 최대 차이 {diff(l0, l1):.2e}")
    print(f"  병합 모델 파라미터 {n_params:,}개(원본과 같음), 저장 {size:.0f}MiB, 걸린 시간 {secs:.0f}초")
    rec["final"] = {"ppl_before": p0, "ppl_after": p1, "logits_diff": diff(l0, l1), "params": n_params,
                    "size_mib": round(size)}
    OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
