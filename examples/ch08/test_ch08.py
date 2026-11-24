# 제08장 검증 테스트
#
# 실행: python -m pytest examples/ch08 -q
# 토크나이저 파일이 이미 내려받아져 있을 때만 실행한다.

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokenize_compare import SENTENCES, load_tokenizer, split_three_ways  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    try:
        return load_tokenizer(local_only=True)
    except Exception:
        pytest.skip("토크나이저가 아직 내려받아지지 않음(tokenize_compare.py를 먼저 실행)")


def test_vocab_sizes(tok):
    assert tok.vocab_size == 151_643
    assert len(tok) == 151_665          # 일반 토큰 + 특수 토큰
    assert len(tok) <= 151_936          # 모델 Embedding 행 수(config.json의 vocab_size) 이내


def test_special_tokens(tok):
    assert tok.bos_token is None        # Qwen2.5는 BOS를 쓰지 않는다
    assert tok.eos_token == "<|im_end|>" and tok.eos_token_id == 151_645
    assert tok.pad_token == "<|endoftext|>" and tok.pad_token_id == 151_643
    assert tok.pad_token_id != tok.eos_token_id


def test_no_token_added_automatically(tok):
    # 문장 앞뒤에 특수 토큰이 자동으로 붙지 않는다: "Hello" 한 단어 → 토큰 1개
    assert tok.encode("Hello") == [9707]


@pytest.mark.parametrize("text", SENTENCES + ["안녕하세요", "FPGA와 LLM 🙂"])
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_korean_needs_more_tokens_per_character(tok):
    ko, en = (split_three_ways(s, tok) for s in SENTENCES)
    assert ko["서브워드(Qwen)"] / ko["문자"] > en["서브워드(Qwen)"] / en["문자"]
