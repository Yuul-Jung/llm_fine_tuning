# 제21장 핵심 실습: 같은 구조의 Base 모델과 Instruct 모델에 같은 질문을 Greedy로 넣어 비교
#
# 실행: python examples/ch21/base_vs_instruct.py
#
# Base 모델(Qwen2.5-0.5B)은 처음 실행할 때 약 1GB를 내려받는다. 두 모델 모두 Apache 2.0 라이선스.

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from transformers.utils import logging  # noqa: E402

from common.qwen import load_model, load_tokenizer  # noqa: E402

logging.disable_progress_bar()
logging.set_verbosity_error()

BASE_ID = "Qwen/Qwen2.5-0.5B"                                  # 대화 형식 학습 전
BASE_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
QUESTIONS = ["대한민국의 수도는 어디인가요?", "물은 몇 도에서 끓나요?"]
NEW_TOKENS = 30


def load_base(local_only: bool = False):
    return AutoModelForCausalLM.from_pretrained(
        BASE_ID, revision=BASE_REVISION, dtype=torch.float32,
        local_files_only=local_only).eval()


def as_chat(tok, question: str) -> str:
    """Instruct 모델이 학습한 대화 형식(Chat Template)으로 감싼다. 자세한 구조는 제44장."""
    return tok.apply_chat_template([{"role": "user", "content": question}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def greedy(model, tok, text: str, n: int = NEW_TOKENS) -> str:
    """생성 설정을 모두 명시해 고정한다. generation_config.json의 무작위 추출·반복 벌점을 끈다."""
    ids = tok(text, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=n, do_sample=False, repetition_penalty=1.0,
                         temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def run_all(models, tok) -> dict:
    results = {}
    for q in QUESTIONS:
        for form, text in (("평문", q), ("대화 형식", as_chat(tok, q))):
            for name, model in models.items():
                results[(q, form, name)] = greedy(model, tok, text)
    return results


def fingerprint(results: dict) -> str:
    joined = "\n".join(f"{k}|{v}" for k, v in sorted(results.items()))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def main() -> None:
    tok = load_tokenizer()                       # 두 모델은 같은 토크나이저(어휘)를 쓴다
    models = {"Base": load_base(), "Instruct": load_model()}
    print(f"대화 형식 예: {as_chat(tok, QUESTIONS[0])!r}\n")

    first = run_all(models, tok)
    for (q, form, name), answer in first.items():
        if name == "Base":
            print(f"── 질문: {q} / 입력: {form} ──")
        print(f"  {name:<9}{answer.strip()[:60]!r}")
    second = run_all(models, tok)
    print(f"\n두 번 실행한 결과 동일: {first == second}  (지문 {fingerprint(first)})")


if __name__ == "__main__":
    main()
