# 제41장 핵심 실습: FPGA 질의응답 데이터셋을 설계 기준으로 검사하고, 세 가지 데이터 구조로 바꿔 본다
#
# 실행: python examples/ch41/check_dataset.py

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

DATA = ROOT / "datasets" / "fpga_qa" / "fpga_qa.json"
FIELDS = ("id", "category", "instruction", "input", "output")
PLAN = {"개념": 10, "Verilog 문법": 10, "설계 기법": 8, "타이밍": 8, "도구와 흐름": 7, "디버깅": 7}


def load(path: Path = DATA) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def problems(rows: list) -> list:
    """설계 기준 위반 목록. 비어 있으면 통과."""
    out = []
    ids, questions = Counter(r.get("id") for r in rows), Counter(r.get("instruction") for r in rows)
    for r in rows:
        rid = r.get("id", "?")
        for f in FIELDS:
            if f not in r:
                out.append(f"{rid}: '{f}' 항목 없음")
            elif f != "input" and not str(r[f]).strip():
                out.append(f"{rid}: '{f}' 비어 있음")
        if r.get("category") not in PLAN:
            out.append(f"{rid}: 설계표에 없는 분류 '{r.get('category')}'")
        out_text = r.get("output", "").strip()
        if not out_text.endswith("다."):
            out.append(f"{rid}: 답변이 '~다.'로 끝나지 않음")
        if "아래 코드" in r.get("instruction", "") and not r.get("input"):
            out.append(f"{rid}: 코드를 가리키는 질문인데 input이 비어 있음")
    out += [f"{k}: id 중복 {v}회" for k, v in ids.items() if v > 1]
    out += [f"질문 중복 {v}회: {k[:20]}" for k, v in questions.items() if v > 1]
    counts = Counter(r.get("category") for r in rows)
    out += [f"분류 '{c}' {counts[c]}개(목표 {n}개)" for c, n in PLAN.items() if counts[c] != n]
    return out


def to_structures(r: dict) -> dict:
    """같은 샘플의 세 가지 구조(02절)"""
    prompt = r["instruction"] + (f"\n\n{r['input']}" if r["input"] else "")
    return {
        "instruction": {k: r[k] for k in ("instruction", "input", "output")},
        "prompt_completion": {"prompt": prompt, "completion": r["output"]},
        "messages": {"messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": r["output"]}]},
    }


def main() -> None:
    rows = load()
    print(f"── 1. 설계 기준 검사({len(rows)}개) ──")
    found = problems(rows)
    print(f"  위반 {len(found)}건" + "".join(f"\n    {p}" for p in found))

    print("\n── 2. 설계표 대비 분류별 개수 ──")
    counts = Counter(r["category"] for r in rows)
    for c, n in PLAN.items():
        print(f"  {c:<12}{counts[c]:>3} / {n}")
    print(f"  코드가 있는 샘플(input 사용) {sum(1 for r in rows if r['input'])}개")

    print("\n── 3. 길이(글자 수, 토큰 수) ──")
    try:
        from common.qwen import load_tokenizer
        tok = load_tokenizer()
        q = [len(tok(r["instruction"] + r["input"]).input_ids) for r in rows]
        a = [len(tok(r["output"]).input_ids) for r in rows]
        print(f"  질문 토큰 평균 {sum(q) / len(q):.0f}, 최대 {max(q)} / 답변 토큰 평균 {sum(a) / len(a):.0f}, "
              f"최대 {max(a)}, 전체 {sum(q) + sum(a):,}")
    except Exception as e:
        print(f"  토크나이저 없음({type(e).__name__}), 글자 수만 계산")
    chars = [len(r["output"]) for r in rows]
    print(f"  답변 글자 평균 {sum(chars) / len(chars):.0f}, 최소 {min(chars)}, 최대 {max(chars)}")

    print("\n── 4. 같은 샘플(fpga-013)의 세 가지 구조 ──")
    sample = next(r for r in rows if r["id"] == "fpga-013")
    for name, s in to_structures(sample).items():
        print(f"  [{name}]\n  {json.dumps(s, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
