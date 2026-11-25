# 제42장 핵심 실습: 제41장 데이터를 JSONL로 저장하고 Hugging Face Datasets로 불러 분할·변환·저장·복원한다
#
# 실행: python examples/ch42/hf_datasets.py
#
# 만드는 파일: datasets/fpga_qa/fpga_qa.jsonl(저장소에 포함), models/fpga_qa_v1(DatasetDict, 저장소 제외)

import csv
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent, HERE.parent / "ch41"):
    sys.path.insert(0, str(p))

from datasets import DatasetDict, load_dataset, load_from_disk  # noqa: E402

from check_dataset import load, to_structures  # noqa: E402     (제41장)

JSONL = ROOT / "datasets" / "fpga_qa" / "fpga_qa.jsonl"
SAVED = ROOT / "models" / "fpga_qa_v1"
SEED = 0


def write_jsonl(rows: list, path: Path = JSONL) -> None:
    """한 줄에 샘플 하나. 줄바꿈은 JSON 안에서 \\n으로 바뀌어 줄 구분과 섞이지 않는다."""
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(jsonl: Path = JSONL) -> DatasetDict:
    ds = load_dataset("json", data_files=str(jsonl), split="train")
    first = ds.train_test_split(test_size=10, seed=SEED)                 # 40 / 10
    second = first["test"].train_test_split(test_size=5, seed=SEED)      # 10 → 5 / 5
    return DatasetDict(train=first["train"], validation=second["train"], test=second["test"])


def add_messages(example: dict) -> dict:
    return {"messages": to_structures(example)["messages"]["messages"]}


def main() -> None:
    rows = load()
    print("── 1. CSV와 JSON, JSONL ──")
    sample = next(r for r in rows if "\n" in r["input"])
    buf = io.StringIO()
    csv.DictWriter(buf, fieldnames=list(sample)).writerows([sample])
    print(f"  CSV 한 샘플이 차지하는 줄 수: {len(buf.getvalue().splitlines())}  (코드 속 줄바꿈 때문에 여러 줄)")
    write_jsonl(rows)
    lines = JSONL.read_text(encoding="utf-8").splitlines()
    print(f"  JSONL 줄 수 {len(lines)} = 샘플 수 {len(rows)}, 파일 {JSONL.stat().st_size:,}바이트")
    print(f"  첫 줄 앞부분: {lines[0][:70]}…")

    print("\n── 2. Dataset과 DatasetDict ──")
    dd = build()
    print(f"  {dd}".replace("\n", "\n  "))
    print(f"  Features: {dd['train'].features}")

    print("\n── 3. map·filter·select ──")
    dd = dd.map(add_messages)
    print(f"  map 후 열: {dd['train'].column_names}")
    code_only = dd["train"].filter(lambda e: e["input"] != "")
    print(f"  filter(코드가 있는 샘플): train {len(code_only)}개 → {list(code_only['id'])}")
    small = dd["train"].select(range(3))
    print(f"  select(처음 3개): {list(small['id'])}")
    cats = list(dd["train"]["category"])
    by_cat = {c: cats.count(c) for c in sorted(set(cats))}
    print(f"  train 분류별 개수: {by_cat}")

    print("\n── 4. 저장과 복원 ──")
    dd.save_to_disk(str(SAVED))
    back = load_from_disk(str(SAVED))
    same = all(back[s].to_list() == dd[s].to_list() for s in dd)
    print(f"  분할별 샘플 수 {[(s, len(back[s])) for s in back]}")
    print(f"  복원 결과가 저장 전과 완전히 같음: {same}")
    print(f"  분할 id(같은 seed면 언제나 같다): validation {list(back['validation']['id'])}, test {list(back['test']['id'])}")


if __name__ == "__main__":
    main()
