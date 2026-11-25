# 제43장 실습 2: 미리 적은 합격 기준을 읽고, 채점 함수를 정답과 나쁜 답으로 시험한다
#
# 실행: python examples/ch43/criteria.py

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CRITERIA = ROOT / "configs" / "eval_criteria.yaml"


def load_criteria(path: Path = CRITERIA) -> dict:
    c = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert {"domain_ppl", "keyword_hit", "format_ok", "general_ppl"} <= set(c["metrics"])
    return c


def keyword_hit(answer: str, keywords: list) -> float:
    a = answer.replace(" ", "").lower()
    return sum(k.replace(" ", "").lower() in a for k in keywords) / len(keywords)


def sentences(answer: str) -> list:
    return [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s]


def format_ok(answer: str, max_sentences: int) -> bool:
    return answer.strip().endswith("다.") and len(sentences(answer)) <= max_sentences


def judge(c: dict, *, domain_ratio: float, general_ratio: float, hits: list, fmts: list) -> dict:
    m = c["metrics"]
    return {
        "domain_ppl": domain_ratio <= m["domain_ppl"]["max_ratio"],
        "keyword_hit": sum(hits) / len(hits) >= m["keyword_hit"]["min_mean"],
        "format_ok": sum(fmts) / len(fmts) >= m["format_ok"]["min_rate"],
        "general_ppl": general_ratio <= m["general_ppl"]["max_ratio"],
    }


def main() -> None:
    c = load_criteria()
    rows = {r["id"]: r for r in json.loads((ROOT / c["data"]["source"]).read_text(encoding="utf-8"))}
    split = json.loads((ROOT / c["data"]["split"]).read_text(encoding="utf-8"))["split"]
    test = [i for i, s in split.items() if s == "test"]
    max_s = c["metrics"]["format_ok"]["max_sentences"]
    assert sorted(test) == sorted(c["keywords"]), "핵심어는 test 문항 전체에 있어야 한다"

    print(f"기준 파일 version {c['version']}, 지표 {list(c['metrics'])}")
    for k, v in c["metrics"].items():
        print(f"  {k:<12}{v['pass']}")

    bad = {"fpga-010": "배치는 회로를 놓는 것", "fpga-031": "임계 경로는 중요한 경로입니다. 잘 설계하세요!"}
    print("\n문항        정답 핵심어  정답 형식   나쁜 답 핵심어  나쁜 답 형식")
    for i in sorted(test):
        ref, kw = rows[i]["output"], c["keywords"][i]
        b = bad.get(i, "잘 모르겠어요")
        print(f"  {i}  {keyword_hit(ref, kw):>8.2f}  {str(format_ok(ref, max_s)):>8}"
              f"  {keyword_hit(b, kw):>12.2f}  {str(format_ok(b, max_s)):>10}")

    hits = [keyword_hit(rows[i]["output"], c["keywords"][i]) for i in test]
    fmts = [format_ok(rows[i]["output"], max_s) for i in test]
    print("\n정답을 모델의 답으로 가정한 판정(PPL 비율은 예시 값 0.75, 1.04):")
    print(" ", judge(c, domain_ratio=0.75, general_ratio=1.04, hits=hits, fmts=fmts))
    print("일반 PPL이 15% 오른 경우:")
    print(" ", judge(c, domain_ratio=0.75, general_ratio=1.15, hits=hits, fmts=fmts))


if __name__ == "__main__":
    main()
