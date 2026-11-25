# 제42장 검증 테스트
#
# 실행: python -m pytest examples/ch42 -q

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch41"):
    sys.path.insert(0, str(p))

pytest.importorskip("datasets")

from check_dataset import load  # noqa: E402
from hf_datasets import add_messages, build, write_jsonl  # noqa: E402


def test_jsonl_roundtrip(tmp_path):
    rows = load()
    path = tmp_path / "a.jsonl"
    write_jsonl(rows, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 50 and [json.loads(x) for x in lines] == rows


def test_split_is_disjoint_and_reproducible(tmp_path):
    path = tmp_path / "a.jsonl"
    write_jsonl(load(), path)
    a, b = build(path), build(path)
    ids = {s: set(a[s]["id"]) for s in a}
    assert [len(a[s]) for s in ("train", "validation", "test")] == [40, 5, 5]
    assert not (ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"])
    assert all(list(a[s]["id"]) == list(b[s]["id"]) for s in a)


def test_map_filter_select_and_save(tmp_path):
    from datasets import load_from_disk
    path = tmp_path / "a.jsonl"
    write_jsonl(load(), path)
    dd = build(path).map(add_messages)
    assert dd["train"][0]["messages"][1]["content"] == dd["train"][0]["output"]
    assert len(dd["train"].filter(lambda e: e["input"] != "")) <= 5
    assert len(dd["train"].select(range(3))) == 3
    dd.save_to_disk(str(tmp_path / "saved"))
    back = load_from_disk(str(tmp_path / "saved"))
    assert all(back[s].to_list() == dd[s].to_list() for s in dd)


def test_committed_jsonl_matches_source():
    jsonl = HERE.parents[1] / "datasets" / "fpga_qa" / "fpga_qa.jsonl"
    rows = [json.loads(x) for x in jsonl.read_text(encoding="utf-8").splitlines()]
    assert rows == load()                       # 원본(json)과 변환본(jsonl)이 어긋나지 않음
