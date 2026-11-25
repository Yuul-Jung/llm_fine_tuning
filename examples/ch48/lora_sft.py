# 제48장 핵심 실습: 분할·템플릿·Label Masking·SFTTrainer·LoRA를 하나의 학습 스크립트로 조립한다
#
# 실행: python examples/ch48/lora_sft.py scope   (Qwen2-Tiny, Target Modules 세 범위 비교, CPU 약 8분)
#       python examples/ch48/lora_sft.py domain  (Qwen2.5-0.5B FP32 LoRA 도메인 학습과 평가, CPU 약 15분)
#
# 만드는 파일: examples/ch48/scope.json, examples/ch48/domain.json, models/ch48_lora/(Adapter)

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent, HERE.parent / "ch46", HERE.parent / "ch47"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers.utils import logging  # noqa: E402

from baseline import measure  # noqa: E402                         (제47장)
from sft_loop import SFTDataset, setup, split_rows, validate  # noqa: E402   (제46장)
from sft_trainer import SAME_AS_CH46, to_hf_dataset  # noqa: E402  (제47장)

logging.disable_progress_bar()
logging.set_verbosity_error()
ADAPTER = ROOT / "models" / "ch48_lora"
SCOPES = {
    "Attention q·v": ["q_proj", "v_proj"],
    "Attention 전체": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "모든 Linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def lora_config(targets: list, r: int = 8) -> LoraConfig:
    return LoraConfig(r=r, lora_alpha=2 * r, target_modules=targets, lora_dropout=0.0,
                      bias="none", task_type="CAUSAL_LM")


def train_lora(model, tok, tr, va, targets: list, **overrides):
    """SFTTrainer가 peft_config를 받아 LoRA를 붙이고 학습한다. 학습된 PEFT 모델과 기록을 돌려준다."""
    from trl import SFTConfig, SFTTrainer
    torch.manual_seed(0)                                           # LoRA A 초기값 고정
    cfg = {**SAME_AS_CH46, "disable_tqdm": True, **overrides}
    trainer = SFTTrainer(model=model, args=SFTConfig(output_dir="/tmp/ch48_run", use_cpu=True, **cfg),
                         processing_class=tok, peft_config=lora_config(targets),
                         train_dataset=to_hf_dataset(tr), eval_dataset=to_hf_dataset(va))
    trainer.train()
    m = trainer.model
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return m, n_train, trainer.state.log_history


def curve(history: list) -> list:
    return [(r["step"], r["eval_loss"]) for r in history if "eval_loss" in r]


def run_scope() -> None:
    print("── Target Modules 범위 비교(Qwen2-Tiny, r 8, lr 2e-3, 30단계) ──")
    results = {}
    for name, targets in SCOPES.items():
        tok, model, tr, va = setup()
        before = validate(model, va)["val_loss"]
        t0 = time.time()
        m, n, hist = train_lora(model, tok, tr, va, targets, learning_rate=2e-3)
        results[name] = {"targets": targets, "trainable": n, "val_before": before,
                         "val_after": validate(m, va)["val_loss"], "curve": curve(hist),
                         "seconds": round(time.time() - t0)}
        r = results[name]
        print(f"  {name:<14}학습 파라미터 {n:>8,}  val {r['val_before']:.3f} → {r['val_after']:.3f}  ({r['seconds']}초)")
    (HERE / "scope.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def run_domain() -> None:
    from common.qwen import load_model, load_tokenizer
    from criteria import judge, load_criteria
    tok = load_tokenizer()
    base = json.loads((ROOT / "examples/ch47/baseline.json").read_text(encoding="utf-8"))
    torch.manual_seed(0)
    model = load_model()
    tr, va = SFTDataset(tok, split_rows("train")), SFTDataset(tok, split_rows("validation"))
    print("── Qwen2.5-0.5B FP32 LoRA(모든 Linear, r 8, lr 2e-4, 30단계) ──")
    t0 = time.time()
    m, n, hist = train_lora(model, tok, tr, va, SCOPES["모든 Linear"], learning_rate=2e-4)
    secs = round(time.time() - t0)
    print(f"  학습 파라미터 {n:,}개, 학습 {secs}초")
    print("  단계   train   val(샘플 평균)")
    train = {r["step"]: r["loss"] for r in hist if "loss" in r}
    for s, v in curve(hist):
        print(f"  {s:>4}  {train[s]:.3f}   {v:.3f}  " + "█" * int(v * 20))
    m.save_pretrained(ADAPTER)
    m.eval()
    after = measure(tok, m)
    c = load_criteria()
    hits, fmts = list(after["keyword_hit"].values()), list(after["format_ok"].values())
    verdict = judge(c, domain_ratio=after["domain_ppl"] / base["domain_ppl"],
                    general_ratio=after["general_ppl"] / base["general_ppl"], hits=hits, fmts=fmts)
    print("\n── 제43장 기준으로 판정(학습 전 → 학습 후) ──")
    rows = [("도메인 PPL", base["domain_ppl"], after["domain_ppl"], "domain_ppl"),
            ("일반 PPL", base["general_ppl"], after["general_ppl"], "general_ppl"),
            ("핵심어 적중률", sum(base["keyword_hit"].values()) / 6, sum(hits) / len(hits), "keyword_hit"),
            ("형식 준수율", sum(base["format_ok"].values()) / 6, sum(fmts) / len(fmts), "format_ok")]
    for label, b, a, key in rows:
        print(f"  {label:<10}{b:>7.2f} → {a:>6.2f}   {'합격' if verdict[key] else '불합격'}")
    print("\n── 응답 비교(Greedy, 최대 160토큰, 앞부분) ──")
    for i in after["answers"]:
        print(f"  [{i}] 전: {base['answers'][i][:46].replace(chr(10), ' ')}…")
        print(f"  {'':>10}후: {after['answers'][i][:46].replace(chr(10), ' ')}…")
    out = {"trainable": n, "seconds": secs, "curve": curve(hist), "train": train,
           "after": after, "verdict": verdict}
    (HERE / "domain.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    {"scope": run_scope, "domain": run_domain}[sys.argv[1]]()
