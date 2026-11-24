# 제21장 검증 테스트
#
# 실행: python -m pytest examples/ch21 -q

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common.fixed_data import SEQ_LEN, fingerprint, load_blocks, make_blocks, read_lines  # noqa: E402

# 제21장에서 기록한 기준 지문. 데이터나 토크나이저가 바뀌면 이 테스트가 알려 준다.
TRAIN_FP, EVAL_FP = "568b36495839b8b1", "1d8635627d660631"


@pytest.fixture(scope="module")
def tok():
    try:
        from common.qwen import load_tokenizer
        return load_tokenizer(True)
    except Exception:
        pytest.skip("Qwen 토크나이저가 아직 내려받아지지 않음(fixed_dataset.py를 먼저 실행)")


def test_make_blocks_drops_tail():
    blocks = make_blocks(list(range(10)), seq_len=4)
    assert blocks.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_eval_sentences_not_in_train():
    assert not set(read_lines("eval")) & set(read_lines("train"))


def test_fixed_blocks_and_fingerprint(tok):
    train, ev = load_blocks(tok, "train"), load_blocks(tok, "eval")
    assert train.shape == (25, SEQ_LEN) and ev.shape == (5, SEQ_LEN)
    assert fingerprint(train) == TRAIN_FP and fingerprint(ev) == EVAL_FP


def test_sentences_end_with_endoftext(tok):
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    flat = load_blocks(tok, "train").flatten().tolist()
    assert eot == 151643 and flat.count(eot) >= 25        # 블록마다 평균 두 문장 이상


def test_greedy_is_reproducible(tok):
    try:
        from base_vs_instruct import greedy, load_base
        base = load_base(local_only=True)
    except Exception:
        pytest.skip("Base 모델이 아직 내려받아지지 않음(base_vs_instruct.py를 먼저 실행)")
    a = greedy(base, tok, "봄이 오면", n=8)
    b = greedy(base, tok, "봄이 오면", n=8)
    assert a == b and len(a) > 0
