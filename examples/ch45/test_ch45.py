# 제45장 검증 테스트
#
# 실행: python -m pytest examples/ch45 -q

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch41", HERE.parent / "ch44"):
    sys.path.insert(0, str(p))

import pytest  # noqa: E402

from check_dataset import load  # noqa: E402
from label_mask import IGNORE, build_example, mask_by_pad_id, pad_batch  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    from common.qwen import load_tokenizer
    return load_tokenizer()


def test_only_answer_and_end_are_targets(tok):
    end = tok.convert_tokens_to_ids("<|im_end|>")
    for r in load():
        e = build_example(tok, r)
        target = [t for t in e["labels"] if t != IGNORE]
        assert target[-1] == end and target.count(end) == 1          # 끝 표시는 하나, 맨 끝
        assert tok.decode(target[:-1]) == r["output"]                 # 나머지는 정확히 정답
        assert len(e["input_ids"]) == len(e["labels"]) == len(e["attention_mask"])


def test_padding_is_excluded(tok):
    rows = load()
    b = pad_batch([build_example(tok, rows[2]), build_example(tok, rows[9])], tok.pad_token_id)
    pad = b["attention_mask"] == 0
    assert pad.any() and (b["labels"][pad] == IGNORE).all() and (b["input_ids"][pad] == tok.pad_token_id).all()


def test_pad_equal_eos_loses_end_token(tok):
    rows = load()
    b = pad_batch([build_example(tok, rows[2]), build_example(tok, rows[9])], tok.eos_token_id)
    assert int((mask_by_pad_id(b, tok.eos_token_id) == tok.eos_token_id).sum()) == 0
    b = pad_batch([build_example(tok, rows[2]), build_example(tok, rows[9])], tok.pad_token_id)
    assert int((mask_by_pad_id(b, tok.pad_token_id) == tok.eos_token_id).sum()) == 2


def test_truncation_counts(tok):
    rows = load()
    assert all(build_example(tok, r)["full_len"] <= 256 for r in rows)
    short = [build_example(tok, r, 128) for r in rows]
    assert all(len(e["input_ids"]) <= 128 for e in short)
    assert sum(1 for e in short if e["full_len"] > 128) == 44
