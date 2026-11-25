# 제50장 핵심 실습: 오답 유형을 나누고, 데이터를 보강해 다시 학습한 뒤, 확정 모델을 test로 한 번만 평가한다
#
# 실행: python examples/ch50/improve.py   (교재 검증 환경 CPU 약 40분, 중간 결과는 runs.json에 저장되어 이어서 실행 가능)
#
# 만드는 파일: examples/ch50/runs.json, examples/ch50/final_test.json, models/ch50/(Adapter)

import gc
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent, HERE.parent / "ch41", HERE.parent / "ch43", HERE.parent / "ch46",
          HERE.parent / "ch47", HERE.parent / "ch48"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402
from transformers.utils import logging  # noqa: E402

from baseline import measure  # noqa: E402                    (제47장)
from check_dataset import load  # noqa: E402                   (제41장)
from clean_split import NGRAM_T, jaccard, ngrams  # noqa: E402  (제43장)
from criteria import judge, load_criteria  # noqa: E402        (제43장)
from lora_sft import SCOPES, lora_config  # noqa: E402         (제48장)
from sft_loop import SFTDataset, split_rows  # noqa: E402      (제46장)
from sft_trainer import SAME_AS_CH46, to_hf_dataset, train_device  # noqa: E402   (제47장)

logging.disable_progress_bar()
logging.set_verbosity_error()
RUNS = HERE / "runs.json"
FINAL = HERE / "final_test.json"
MODELS = ROOT / "models" / "ch50"

# 제48장 LoRA의 validation 답을 사람이 읽고 분류한 결과(유형, 근거, train에 관련 지식이 있는가)
ERRORS = {
    "fpga-005": ("사실 오류", "DSP를 '디지털 소규모 프로그래밍'으로 풀어 지어냄", False),
    "fpga-019": ("사실 오류", "generate를 '합성할 때 재사용하는 블록'으로 설명", False),
    "fpga-024": ("질문 오해", "몇까지 세야 하는지 계산하지 않고 '어렵다'고 답함", False),
    "fpga-033": ("사실 오류", "메타안정성을 '같은 값으로 유지되는 기능'으로 설명", True),
    "fpga-041": ("사실 오류", "ILA를 설계 파일 비교 도구처럼 설명, 권유형으로 끝남", False),
    "fpga-049": ("질문 오해", "래치를 찾지 못하고 'q가 d보다 클 때'를 문제로 짚음", True),
}
RELATED = {"fpga-033": "fpga-032(클럭 도메인 사이 동기화)", "fpga-049": "fpga-021(래치가 생기는 원인)"}


def augmented_rows() -> list:
    """train 38개 + 질문만 바꾼 변형 38개(답·입력·분류는 원본)"""
    src = {r["id"]: r for r in load()}
    aug = json.loads((ROOT / "datasets/fpga_qa/augment_v1.json").read_text(encoding="utf-8"))["rows"]
    return split_rows("train") + [dict(src[a["source_id"]], id=a["id"], instruction=a["instruction"]) for a in aug]


def leak_check(rows: list) -> float:
    """보강 데이터가 validation·test 문항과 비슷하지 않은지(제43장 기준)"""
    held = split_rows("validation") + split_rows("test")
    g = {r["id"]: ngrams(r["instruction"] + " " + r["input"]) for r in rows + held}
    return max(jaccard(g[a["id"]], g[h["id"]]) for a in rows for h in held)


def history(name: str) -> tuple:
    """Trainer가 Checkpoint에 남긴 기록에서 학습·검증 손실을 읽는다"""
    log = json.loads((MODELS / name / "checkpoint-30" / "trainer_state.json").read_text())["log_history"]
    return [(r["step"], r["loss"]) for r in log if "loss" in r], \
        [(r["step"], r["eval_loss"]) for r in log if "eval_loss" in r]


def train_run(name: str, rows: list, tok) -> tuple:
    """30단계 학습, 15단계와 30단계 Adapter를 저장한다(15단계 = 제48장의 검증 손실 최저 지점)"""
    from common.qwen import load_model
    from trl import SFTConfig, SFTTrainer
    if (MODELS / name / "checkpoint-30" / "trainer_state.json").exists():
        return history(name)                                     # 이미 학습했으면 기록만 읽는다
    torch.manual_seed(0)
    model = load_model()
    tr, va = SFTDataset(tok, rows), SFTDataset(tok, split_rows("validation"))
    cfg = {**SAME_AS_CH46, "learning_rate": 2e-4, "save_strategy": "steps", "save_steps": 15,
           "disable_tqdm": True, "use_cpu": train_device(6144) == "cpu"}   # FP32 0.5B: GT 1030이면 CPU
    trainer = SFTTrainer(model=model, args=SFTConfig(output_dir=str(MODELS / name), **cfg),
                         processing_class=tok, peft_config=lora_config(SCOPES["모든 Linear"]),
                         train_dataset=to_hf_dataset(tr), eval_dataset=to_hf_dataset(va))
    trainer.train()
    del trainer, model
    gc.collect()                                                 # 메모리 7GB 환경: 다음 모델 전에 비운다
    return history(name)


def load_adapter(name: str, step: int):
    from common.qwen import load_model
    from peft import PeftModel
    return PeftModel.from_pretrained(load_model(), MODELS / name / f"checkpoint-{step}").eval()


def summary(m: dict, base: dict, c: dict) -> dict:
    hits, fmts = list(m["keyword_hit"].values()), list(m["format_ok"].values())
    v = judge(c, domain_ratio=m["domain_ppl"] / base["domain_ppl"],
              general_ratio=m["general_ppl"] / base["general_ppl"], hits=hits, fmts=fmts)
    return {"domain_ppl": m["domain_ppl"], "general_ppl": m["general_ppl"], "keyword": sum(hits) / len(hits),
            "format": sum(fmts) / len(fmts), "passed": sum(v.values()), "verdict": v}


def main() -> None:
    from common.qwen import load_tokenizer
    tok = load_tokenizer()
    c = load_criteria()
    base = json.loads((ROOT / "examples/ch47/baseline.json").read_text(encoding="utf-8"))

    print("── 1. 오답 유형표(제48장 LoRA, validation 6문항) ──")
    for i, (kind, why, covered) in ERRORS.items():
        print(f"  {i}  {kind:<6} train 관련 지식 {'있음' if covered else '없음'}  {why}")
    kinds = {k: sum(1 for v in ERRORS.values() if v[0] == k) for k in sorted({v[0] for v in ERRORS.values()})}
    print(f"  유형별: {kinds}, train에 관련 지식이 있는 문항: {[f'{k} ← {v}' for k, v in RELATED.items()]}")

    rows = augmented_rows()
    print(f"\n── 2. 데이터 보강: train {len(split_rows('train'))} + 질문 변형 {len(rows) - 38} = {len(rows)}개 ──")
    print(f"  보강 데이터와 validation·test의 최대 3-gram 유사도 {leak_check(rows):.2f} < {NGRAM_T}")

    runs = json.loads(RUNS.read_text(encoding="utf-8")) if RUNS.exists() else {}
    for name, data in (("원본", split_rows("train")), ("보강", rows)):
        if name not in runs:
            train, val = train_run(name, data, tok)
            runs[name] = {"train": train, "val": val}
            RUNS.write_text(json.dumps(runs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        for step in (15, 30):
            key = f"{name}-{step}"
            if key not in runs:
                model = load_adapter(name, step)
                runs[key] = summary(measure(tok, model), base, c)
                del model
                gc.collect()
            RUNS.write_text(json.dumps(runs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print("\n── 3. 과적합 판정: 학습·검증 손실(샘플 평균) ──")
    for name in ("원본", "보강"):
        tr, va = dict(runs[name]["train"]), dict(runs[name]["val"])
        best = min(va, key=va.get)
        print(f"  {name}: " + "  ".join(f"{s}단계 {tr[s]:.2f}/{va[s]:.2f}" for s in sorted(va, key=int))
              + f"  → 검증 최저 {best}단계")

    print("\n── 4. 후보 비교(validation, 제43장 기준) ──")
    print("  후보      도메인PPL  일반PPL  핵심어  형식   합격")
    cands = ["원본-15", "원본-30", "보강-15", "보강-30"]
    for k in cands:
        r = runs[k]
        print(f"  {k:<8}{r['domain_ppl']:>9.2f}{r['general_ppl']:>9.2f}{r['keyword']:>8.2f}{r['format']:>6.2f}"
              f"   {r['passed']}/4")
    chosen = max(cands, key=lambda k: (runs[k]["passed"], runs[k]["keyword"], -runs[k]["domain_ppl"]))
    print(f"  선택 규칙: 합격 수 → 핵심어 → 도메인 PPL 순. 확정 모델: {chosen}")

    print("\n── 5. 최종 test 평가(확정 모델, 한 번만) ──")
    if not FINAL.exists():
        name, step = chosen.split("-")
        m = measure(tok, load_adapter(name, int(step)), split="test")
        FINAL.write_text(json.dumps({"chosen": chosen, "result": m}, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    m = final["result"]
    hits, fmts = list(m["keyword_hit"].values()), list(m["format_ok"].values())
    v = judge(c, domain_ratio=m["domain_ppl"] / base["domain_ppl"],
              general_ratio=m["general_ppl"] / base["general_ppl"], hits=hits, fmts=fmts)
    print(f"  {final['chosen']}: 핵심어 {sum(hits) / len(hits):.2f}, 형식 {sum(fmts) / len(fmts):.2f}"
          f"(test), 도메인 PPL {m['domain_ppl']:.2f}, 일반 PPL {m['general_ppl']:.2f}")
    print(f"  판정 {v} → {sum(v.values())}/4")
    for i, a in m["answers"].items():
        print(f"  [{i}] 핵심어 {m['keyword_hit'][i]:.2f}  {a[:46].replace(chr(10), ' ')}…")


if __name__ == "__main__":
    main()
