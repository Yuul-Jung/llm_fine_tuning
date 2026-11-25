# 제47장 실습 2: 학습 전에 Qwen2.5-0.5B-Instruct의 기준값을 제43장 지표로 기록한다
#
# 실행: python examples/ch47/baseline.py   (CPU 약 3분)
#
# 만드는 파일: examples/ch47/baseline.json

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "ch25", HERE.parent / "ch43", HERE.parent / "ch44",
          HERE.parent / "ch45", HERE.parent / "ch46"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from chat_template import render  # noqa: E402                 (제44장)
from common.fixed_data import load_blocks  # noqa: E402
from common.qwen import load_model, load_tokenizer  # noqa: E402
from criteria import format_ok, keyword_hit, load_criteria  # noqa: E402   (제43장)
from label_mask import chat  # noqa: E402                     (제45장)
from sft_loop import SFTDataset, split_rows, validate  # noqa: E402   (제46장)
from trainer import eval_loss  # noqa: E402                   (제25장)

OUT = HERE / "baseline.json"
GEN = {"max_new_tokens": 160, "do_sample": False}


def answer(tok, model, r: dict) -> str:
    text = render(tok, chat(r)[:2], gen=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, **GEN, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def measure(tok, model, split: str = "validation") -> dict:
    """제43장 기준 파일의 네 지표를 잰다(학습 전후에 같은 함수를 쓴다).

    핵심어·형식은 split의 문항으로 생성해 잰다. 학습 중 판정은 validation, test는 마지막에 한 번만.
    """
    c = load_criteria()
    keywords = c["dev_keywords"] if split == "validation" else c["keywords"]
    va = SFTDataset(tok, split_rows("validation"))
    general = math.exp(eval_loss(model, load_blocks(tok, "eval"), str(model.device)))   # 모델이 있는 장치에서
    max_s = c["metrics"]["format_ok"]["max_sentences"]
    answers = {r["id"]: answer(tok, model, r) for r in split_rows(split)}
    return {
        "domain_ppl": validate(model, va)["val_ppl"],
        "general_ppl": general,
        "split": split,
        "keyword_hit": {i: keyword_hit(a, keywords[i]) for i, a in answers.items()},
        "format_ok": {i: format_ok(a, max_s) for i, a in answers.items()},
        "answers": answers,
        "generation": GEN,
    }


def main() -> None:
    torch.manual_seed(0)
    tok, model = load_tokenizer(), load_model()
    m = measure(tok, model)
    OUT.write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    hits, fmts = list(m["keyword_hit"].values()), list(m["format_ok"].values())
    print("── Qwen2.5-0.5B-Instruct 기준값(학습 전) ──")
    print(f"  도메인 PPL(validation 답 부분)  {m['domain_ppl']:.2f}")
    print(f"  일반 PPL(고정 eval)            {m['general_ppl']:.2f}")
    print(f"  핵심어 적중률(validation 평균)  {sum(hits) / len(hits):.2f}")
    print(f"  형식 준수율(validation)         {sum(fmts) / len(fmts):.2f}")
    print("\n── validation 답(앞부분) ──")
    for i, a in m["answers"].items():
        print(f"  {i}  핵심어 {m['keyword_hit'][i]:.2f}  형식 {m['format_ok'][i]!s:<5}  {a[:44].replace(chr(10), ' ')}…")
    c = load_criteria()["metrics"]
    print("\n── 학습 후 합격선(제43장 기준 파일) ──")
    print(f"  도메인 PPL ≤ {m['domain_ppl'] * c['domain_ppl']['max_ratio']:.2f}, "
          f"일반 PPL ≤ {m['general_ppl'] * c['general_ppl']['max_ratio']:.2f}, "
          f"핵심어 ≥ {c['keyword_hit']['min_mean']:.2f}, 형식 ≥ {c['format_ok']['min_rate']:.2f}")


if __name__ == "__main__":
    main()
