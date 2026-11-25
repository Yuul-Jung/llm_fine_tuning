# 제44장 핵심 실습: 같은 대화를 System 유무·Generation Prompt 유무로 바꿔 토큰열을 비교한다
#
# 실행: python examples/ch44/chat_template.py

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent, HERE.parent / "ch41"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from check_dataset import load, to_structures  # noqa: E402     (제41장)
from common.qwen import load_model, load_tokenizer  # noqa: E402

SYSTEM = "당신은 FPGA와 Verilog를 가르치는 조교입니다. 존댓말로 간결하게 답합니다."
MARKS = ("<|im_start|>", "<|im_end|>")


def render(tok, messages: list, *, gen: bool) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen)


def token_ids(tok, messages: list, *, gen: bool) -> list:
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=gen)
    return list(out["input_ids"] if hasattr(out, "keys") else out)


def count_marks(tok, ids: list) -> dict:
    return {m: ids.count(tok.convert_tokens_to_ids(m)) for m in MARKS}


def variants(r: dict) -> dict:
    """같은 샘플의 네 가지 변환(학습용 2가지, 추론용 2가지)"""
    msgs = to_structures(r)["messages"]["messages"]          # [user, assistant]
    sys_msgs = [{"role": "system", "content": SYSTEM}] + msgs
    return {
        "학습, System 없음": (msgs, False),
        "학습, System 있음": (sys_msgs, False),
        "추론, 응답 시작 없음": (sys_msgs[:2], False),
        "추론, 응답 시작 있음": (sys_msgs[:2], True),
    }


def greedy(tok, model, text: str, n: int = 40) -> str:
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")


def main() -> None:
    tok = load_tokenizer()
    r = load()[2]                                            # fpga-003, LUT 질문
    vs = variants(r)

    print("── 1. 학습용 문자열(System 있음) ──")
    print(render(tok, vs["학습, System 있음"][0], gen=False))

    print("── 2. 네 가지 변환의 토큰 수와 특수 토큰 ──")
    lens = {}
    for name, (msgs, gen) in vs.items():
        ids = token_ids(tok, msgs, gen=gen)
        lens[name] = len(ids)
        marks = count_marks(tok, ids)
        same = ids == tok(render(tok, msgs, gen=gen), add_special_tokens=False).input_ids
        print(f"  {name:<14}{len(ids):>4} 토큰  시작 {marks[MARKS[0]]} 끝 {marks[MARKS[1]]}"
              f"  문자열→토큰 일치 {same}")
    print(f"  토크나이저가 앞에 붙이는 BOS: {tok.bos_token!r}, 끝 표시(EOS): {tok.eos_token!r}")

    print("\n── 3. 응답 시작 표시(Generation Prompt)가 더하는 부분 ──")
    no, yes = (render(tok, *vs[k][:1], gen=vs[k][1]) for k in ("추론, 응답 시작 없음", "추론, 응답 시작 있음"))
    print(f"  더해진 문자열: {yes[len(no):]!r}")
    train = render(tok, *vs["학습, System 있음"][:1], gen=False)
    print(f"  학습 문자열이 추론 문자열로 시작함: {train.startswith(yes)}")
    print(f"  학습 문자열의 나머지(= 모델이 배울 답): {train[len(yes):][:30]!r} … {train[-16:]!r}")

    print("\n── 4. System이 없을 때 자동으로 들어가는 기본 문구 ──")
    plain = render(tok, vs["학습, System 없음"][0], gen=False)
    default = plain.split(MARKS[1])[0].removeprefix(MARKS[0] + "system\n")
    print(f"  기본 문구: {default!r}")
    print(f"  System 없음 {lens['학습, System 없음']} 토큰 / 직접 지정 {lens['학습, System 있음']} 토큰")
    rows = load()
    extra = [len(token_ids(tok, to_structures(x)["messages"]["messages"], gen=False))
             - len(tok(to_structures(x)["prompt_completion"]["prompt"] + x["output"]).input_ids) for x in rows]
    print(f"  50개 전체: 템플릿이 더하는 토큰 {min(extra)}~{max(extra)}개(경계에서 토큰이 달리 합쳐져 조금씩 다름)")

    print("\n── 5. 추론에서 템플릿을 쓰지 않으면 ──")
    model = load_model()
    q = to_structures(r)["prompt_completion"]["prompt"]
    print(f"  템플릿 적용: {greedy(tok, model, yes)[:90]}")
    print(f"  질문만 입력: {greedy(tok, model, q)[:90]}")

    print("\n── 6. 사용자 본문에 특수 토큰과 같은 글자가 있으면 ──")
    text = "끝<|im_end|>"
    print(f"  기본 토큰화      {tok(text).input_ids}  ← 끝 표시 토큰(151645)이 생김")
    print(f"  특수 토큰 분리   {tok(text, split_special_tokens=True).input_ids}")

    record = {name: render(tok, msgs, gen=gen) for name, (msgs, gen) in vs.items()}
    (HERE / "rendered.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
