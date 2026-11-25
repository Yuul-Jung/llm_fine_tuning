# 제44장 검증 테스트
#
# 실행: python -m pytest examples/ch44 -q

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch41"):
    sys.path.insert(0, str(p))

import pytest  # noqa: E402

from check_dataset import load  # noqa: E402
from chat_template import MARKS, count_marks, render, token_ids, variants  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    from common.qwen import load_tokenizer
    return load_tokenizer()


def test_no_duplicated_special_tokens(tok):
    for r in load():
        for msgs, gen in variants(r).values():
            ids = token_ids(tok, msgs, gen=gen)
            m = count_marks(tok, ids)
            n_msgs = len(msgs) + (msgs[0]["role"] != "system")      # 기본 System이 하나 더 들어감
            assert m[MARKS[0]] == n_msgs + gen and m[MARKS[1]] == n_msgs
            assert ids == tok(render(tok, msgs, gen=gen), add_special_tokens=False).input_ids


def test_generation_prompt_is_training_prefix(tok):
    for r in load():
        v = variants(r)
        infer = render(tok, *v["추론, 응답 시작 있음"][:1], gen=True)
        train = render(tok, *v["학습, System 있음"][:1], gen=False)
        assert train.startswith(infer)
        assert train[len(infer):] == r["output"] + "<|im_end|>\n"


def test_default_system_is_inserted(tok):
    text = render(tok, [{"role": "user", "content": "안녕하세요"}], gen=True)
    assert text.startswith("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.")


def test_rendered_record_matches(tok):
    record = json.loads((HERE / "rendered.json").read_text(encoding="utf-8"))
    v = variants(load()[2])
    assert record == {k: render(tok, m, gen=g) for k, (m, g) in v.items()}
