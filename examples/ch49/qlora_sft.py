# 제49장 핵심 실습: Qwen2.5-0.5B NF4 + LoRA를 SFTTrainer로 도메인 학습하고 제48장 LoRA와 비교한다
#
# 실행: python examples/ch49/qlora_sft.py   (교재 검증 환경 CPU 약 25분: 학습 + 평가)
#
# GT 1030 경로: 같은 스크립트가 GPU를 찾으면 cuda로 실행한다(미검증, 아래 예산표 참고).
# 만드는 파일: examples/ch49/qlora.json, models/ch49_qlora/(Adapter)

import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent, HERE.parent / "ch37", HERE.parent / "ch38", HERE.parent / "ch39",
          HERE.parent / "ch43", HERE.parent / "ch46", HERE.parent / "ch47", HERE.parent / "ch48"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402
from transformers.utils import logging  # noqa: E402
from peft import prepare_model_for_kbit_training  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

from baseline import measure  # noqa: E402                        (제47장)
from common.fixed_data import load_blocks  # noqa: E402
from common.qwen import load_tokenizer  # noqa: E402
from criteria import judge, load_criteria  # noqa: E402            (제43장)
from load_4bit import hardware, load_nf4, memory_parts  # noqa: E402   (제37장)
from lora_sft import SCOPES, lora_config  # noqa: E402             (제48장)
from qlora_memory import saved_bytes_of_step  # noqa: E402          (제38장)
from qlora_train import keep_training_layout  # noqa: E402         (제39장)
from sft_loop import SFTDataset, split_rows  # noqa: E402          (제46장)
from sft_trainer import SAME_AS_CH46, to_hf_dataset  # noqa: E402  (제47장)

logging.disable_progress_bar()
logging.set_verbosity_error()
ADAPTER = ROOT / "models" / "ch49_qlora"
MiB = 2 ** 20
# GT 1030 경로 설정: 묶음 1 × 누적 8 = 단계당 8샘플(제48장과 같음), Checkpointing, 최대 256
QLORA = {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 8, "gradient_checkpointing": True,
         "gradient_checkpointing_kwargs": {"use_reentrant": False}, "learning_rate": 2e-4}


class FiniteCheck(TrainerCallback):
    """기록마다 손실·Gradient 노름이 유한한지 센다"""

    def __init__(self):
        self.bad = 0

    def on_log(self, args, state, control, logs=None, **kw):
        for k in ("loss", "grad_norm", "eval_loss"):
            if k in (logs or {}) and not math.isfinite(float(logs[k])):
                self.bad += 1


def build(device: str):
    torch.manual_seed(0)
    model = load_nf4(device)
    parts = memory_parts(model)                        # PEFT로 감싸기 전에 센다(제39장)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    n_layout = keep_training_layout(model) if device == "cpu" else 0
    return model, parts, n_layout


def train_qlora(model, tok, tr, va, device: str, callbacks: list):
    """제48장과 같은 조립. 장치에 따라 use_cpu를 정하고, 점검 Callback을 붙인다."""
    from trl import SFTConfig, SFTTrainer
    torch.manual_seed(0)
    args = SFTConfig(output_dir="/tmp/ch49_run", use_cpu=(device == "cpu"), disable_tqdm=True,
                     **{**SAME_AS_CH46, **QLORA})
    trainer = SFTTrainer(model=model, args=args, processing_class=tok,
                         peft_config=lora_config(SCOPES["모든 Linear"]), callbacks=callbacks,
                         train_dataset=to_hf_dataset(tr), eval_dataset=to_hf_dataset(va))
    trainer.train()
    m = trainer.model
    return m, sum(p.numel() for p in m.parameters() if p.requires_grad), trainer.state.log_history


def budget(model, tok, tr, parts: dict, n_lora: int) -> dict:
    """GT 1030(2GB) 예산: 가중치 + LoRA(FP32 값·Gradient·AdamW 2개) + 가장 긴 샘플의 Activation"""
    longest = max(tr.items, key=lambda e: len(e["input_ids"]))
    ids = torch.tensor([longest["input_ids"]])
    act, _, _ = saved_bytes_of_step(model, ids)
    fixed = load_blocks(tok, "train", seq_len=256)[:1]
    act_fixed, _, _ = saved_bytes_of_step(model, fixed)
    model.zero_grad(set_to_none=True)
    return {"weights": sum(parts.values()), "lora_states": n_lora * 4 * 4, "act_real": act,
            "act_fixed256": act_fixed, "longest": len(longest["input_ids"])}


def main() -> None:
    hw = hardware()
    device = hw["device"]
    tok = load_tokenizer()
    tr, va = SFTDataset(tok, split_rows("train")), SFTDataset(tok, split_rows("validation"))
    print(f"── 1. 4비트 모델과 PEFT(장치 {hw['name']}) ──")
    model, parts, n_layout = build(device)
    print(f"  원본 메모리 {sum(parts.values()) / MiB:.1f}MiB, CPU 추론용 재배치 끔 {n_layout}개")
    start = measure(tok, model)                       # QLoRA의 학습 전 = NF4 기본 모델
    model.train()
    print(f"  NF4 기본 모델: 도메인 PPL {start['domain_ppl']:.2f}, 일반 PPL {start['general_ppl']:.2f}"
          f"(FP32는 17.64, 21.45)")

    check = FiniteCheck()
    print("\n── 2·3. 학습(NF4, 모든 Linear r 8, 묶음 1 × 누적 8, Checkpointing, 30단계) ──")
    t0 = time.time()
    m, n, hist = train_qlora(model, tok, tr, va, device, [check])
    secs = round(time.time() - t0)
    train = {r["step"]: r["loss"] for r in hist if "loss" in r}
    curve = [(r["step"], r["eval_loss"]) for r in hist if "eval_loss" in r]
    print(f"  학습 파라미터 {n:,}개, 학습 {secs}초, NaN·Inf 기록 {check.bad}건")
    print("  단계   train   val(샘플 평균)")
    for s, v in curve:
        print(f"  {s:>4}  {train[s]:.3f}   {v:.3f}")

    b = budget(m, tok, tr, parts, n)
    total = b["weights"] + b["lora_states"] + b["act_real"]
    print("\n── 메모리 예산(GT 1030 2GB 기준, 계산값) ──")
    print(f"  4비트 가중치 등 {b['weights'] / MiB:7.1f}MiB")
    print(f"  LoRA 값·Gradient·AdamW 2개 {b['lora_states'] / MiB:7.1f}MiB")
    print(f"  Activation(실제 데이터 최장 {b['longest']}토큰) {b['act_real'] / MiB:7.1f}MiB"
          f"  / 고정 데이터 256토큰 {b['act_fixed256'] / MiB:.1f}MiB")
    print(f"  합계 {total / MiB:.1f}MiB(CUDA 실행 공간·단편화 여유 제외) < 2048MiB: {total < 2048 * MiB}")

    m.save_pretrained(ADAPTER)                       # 학습 결과부터 저장
    m.eval()
    after = measure(tok, m)
    base = json.loads((ROOT / "examples/ch47/baseline.json").read_text(encoding="utf-8"))
    lora = json.loads((ROOT / "examples/ch48/domain.json").read_text(encoding="utf-8"))
    hits, fmts = list(after["keyword_hit"].values()), list(after["format_ok"].values())
    verdict = judge(load_criteria(), domain_ratio=after["domain_ppl"] / start["domain_ppl"],
                    general_ratio=after["general_ppl"] / start["general_ppl"], hits=hits, fmts=fmts)
    vs_fp32 = judge(load_criteria(), domain_ratio=after["domain_ppl"] / base["domain_ppl"],
                    general_ratio=after["general_ppl"] / base["general_ppl"], hits=hits, fmts=fmts)

    def avg(d):
        return sum(d.values()) / len(d)
    print("\n── 4. LoRA(제48장, FP32)와 QLoRA(NF4) 비교 ──")
    print("  지표        FP32 전  LoRA 후  NF4 전  QLoRA 후  판정(NF4 전 기준 / FP32 전 기준)")
    for label, key, f in (("도메인 PPL", "domain_ppl", None), ("일반 PPL", "general_ppl", None),
                          ("핵심어", "keyword_hit", avg), ("형식", "format_ok", avg)):
        g = f or (lambda x: x)
        mark = {True: "합격", False: "불합격"}
        print(f"  {label:<9}{g(base[key]):>8.2f}{g(lora['after'][key]):>9.2f}{g(start[key]):>8.2f}"
              f"{g(after[key]):>10.2f}   {mark[verdict[key]]} / {mark[vs_fp32[key]]}")
    print("\n── 답 비교(앞부분) ──")
    for i in after["answers"]:
        print(f"  [{i}] LoRA:  {lora['after']['answers'][i][:40].replace(chr(10), ' ')}…")
        print(f"  {'':>10}QLoRA: {after['answers'][i][:40].replace(chr(10), ' ')}…")
    out = {"device": device, "trainable": n, "seconds": secs, "nonfinite": check.bad, "curve": curve,
           "train": train, "budget": b, "start": start, "after": after, "verdict": verdict, "verdict_vs_fp32": vs_fp32}
    (HERE / "qlora.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
