# 제45장 핵심 실습: 샘플 하나의 input_ids와 labels를 나란히 출력해 학습 대상 토큰을 표시한다
#
# 실행: python examples/ch45/label_mask.py

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "ch41", HERE.parent / "ch44"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from chat_template import SYSTEM, render  # noqa: E402           (제44장)
from check_dataset import load, to_structures  # noqa: E402      (제41장)
from common.qwen import load_model, load_tokenizer  # noqa: E402

IGNORE = -100
MAX_LEN = 256                               # GT 1030 기준 최대 시퀀스


def chat(r: dict) -> list:
    return [{"role": "system", "content": SYSTEM}] + to_structures(r)["messages"]["messages"]


def build_example(tok, r: dict, max_len: int = MAX_LEN) -> dict:
    """질문 부분은 -100, 답과 끝 표시만 학습 대상. 끝 표시 뒤의 줄바꿈은 가린다."""
    msgs = chat(r)
    prompt = tok(render(tok, msgs[:2], gen=True), add_special_tokens=False).input_ids
    full = tok(render(tok, msgs, gen=False), add_special_tokens=False).input_ids
    assert full[:len(prompt)] == prompt, "추론 문자열이 학습 문자열의 앞부분이어야 한다"
    end = full.index(tok.convert_tokens_to_ids("<|im_end|>"), len(prompt))
    labels = [IGNORE] * len(prompt) + full[len(prompt):end + 1] + [IGNORE] * (len(full) - end - 1)
    ids, labels = full[:max_len], labels[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels,
            "answer_tokens": end + 1 - len(prompt), "full_len": len(full)}


def pad_batch(examples: list, pad_id: int) -> dict:
    """오른쪽 PAD: 보지 않을 자리(attention 0)이자 채점하지 않을 자리(label -100)"""
    n = max(len(e["input_ids"]) for e in examples)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for e in examples:
        k = n - len(e["input_ids"])
        out["input_ids"].append(e["input_ids"] + [pad_id] * k)
        out["attention_mask"].append(e["attention_mask"] + [0] * k)
        out["labels"].append(e["labels"] + [IGNORE] * k)
    return {key: torch.tensor(v) for key, v in out.items()}


def mask_by_pad_id(batch: dict, pad_id: int) -> torch.Tensor:
    """흔한 방식: PAD 위치를 따로 기록하지 않고, PAD id와 같은 토큰을 모두 -100으로 만든다"""
    labels = batch["labels"].clone()
    labels[batch["input_ids"] == pad_id] = IGNORE
    return labels


def show(tok, e: dict, around: int = 4) -> None:
    ids, labels = e["input_ids"], e["labels"]
    start = labels.index(next(x for x in labels if x != IGNORE))
    keep = set(range(start - around, start + around)) | set(range(len(ids) - around, len(ids)))
    print("  위치  학습  input_id  토큰")
    for i in sorted(keep):
        mark = "●" if labels[i] != IGNORE else "·"
        print(f"  {i:>4}   {mark}   {ids[i]:>7}  {tok.decode([ids[i]])!r}")
        if i == start + around - 1:
            print("   …")


def main() -> None:
    tok = load_tokenizer()
    rows = load()
    r = rows[2]                                             # fpga-003
    e = build_example(tok, r)

    print("── 1. input_ids · attention_mask · labels(fpga-003) ──")
    n_train = sum(x != IGNORE for x in e["labels"])
    print(f"  길이 {len(e['input_ids'])}, attention 1인 자리 {sum(e['attention_mask'])}, 학습 대상 {n_train}"
          f"(답 {n_train - 1} + 끝 표시 1)")
    show(tok, e)

    print("\n── 2. Attention Mask와 Loss Mask(두 샘플을 오른쪽 PAD로 묶음) ──")
    b = pad_batch([e, build_example(tok, rows[9])], tok.pad_token_id)
    for j in range(2):
        a, lab = b["attention_mask"][j], b["labels"][j]
        print(f"  샘플 {j}: 길이 {len(a)}, 보는 자리 {int(a.sum())}, 채점 자리 {int((lab != IGNORE).sum())}, "
              f"PAD {int((a == 0).sum())}")
    print(f"  PAD 토큰 {tok.pad_token!r}(id {tok.pad_token_id}), 끝 표시 {tok.eos_token!r}(id {tok.eos_token_id})")

    print("\n── 3. Completion-only Loss: 질문 부분과 답 부분의 Loss(Qwen2.5 그대로) ──")
    model = load_model()
    ids = torch.tensor([e["input_ids"]])
    with torch.no_grad():
        logits = model(ids).logits[0, :-1]
    tok_loss = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="none")
    target = torch.tensor(e["labels"][1:]) != IGNORE
    print(f"  가린 부분 평균 Loss {tok_loss[~target].mean():.3f}({int((~target).sum())}토큰, 질문과 끝 줄바꿈)")
    print(f"  답 부분 평균 Loss  {tok_loss[target].mean():.3f}({int(target.sum())}토큰)  ← 학습이 줄일 값")
    lab = torch.tensor([e["labels"]])
    print(f"  labels로 계산한 Loss {model(ids, labels=lab).loss.item():.3f}(답 부분 평균과 같아야 함)")

    print("\n── 4. Assistant-only Loss의 조건 ──")
    has = "{% generation %}" in tok.chat_template
    out = tok.apply_chat_template(chat(r), tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
    print(f"  템플릿에 생성 구간 표시 있음: {has}, assistant 마스크의 1 개수: {sum(out['assistant_masks'])}")

    print("\n── 5. 끝 표시와 PAD ──")
    print("  PAD id로 가리는 방식을 썼을 때 학습 대상으로 남는 끝 표시(샘플 2개, 정답은 2개):")
    for name, pad in (("PAD = endoftext(Qwen2.5 Instruct 기본)", tok.pad_token_id),
                      ("PAD = 끝 표시(PAD를 EOS로 지정)", tok.eos_token_id)):
        bb = pad_batch([e, build_example(tok, rows[9])], pad)
        left = int((mask_by_pad_id(bb, pad) == tok.eos_token_id).sum())
        print(f"    {name}: {left}개 → {'정상' if left == 2 else '모델이 멈추는 법을 배우지 못함'}")

    print("\n── 6. 길이 제한과 잘림(50개, 질문 앞부분은 같고 답이 잘림) ──")
    for max_len in (256, 128):
        ex = [build_example(tok, x, max_len) for x in rows]
        cut = [x for x in ex if x["full_len"] > max_len]
        lost = sum(x["answer_tokens"] - sum(t != IGNORE for t in x["labels"]) for x in ex)
        total = sum(x["answer_tokens"] for x in ex)
        no_end = sum(1 for x in ex if tok.eos_token_id not in [t for t in x["labels"] if t != IGNORE])
        print(f"  최대 {max_len}: 잘린 샘플 {len(cut)}개, 잃은 학습 대상 토큰 {lost}(답·끝 표시 토큰의 {lost / total:.1%}), "
              f"끝 표시를 못 배우는 샘플 {no_end}개")
    lens = sorted(build_example(tok, x)["full_len"] for x in rows)
    print(f"  전체 길이: 최소 {lens[0]}, 중앙 {lens[25]}, 최대 {lens[-1]}")


if __name__ == "__main__":
    main()
