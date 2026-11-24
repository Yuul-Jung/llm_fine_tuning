# 제09장 검증 테스트
#
# 실행: python -m pytest examples/ch09 -q

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bpe import decode, encode, merge_symbols, pair_counts, to_words, train_bpe  # noqa: E402

CORPUS = Path(__file__).parents[1].joinpath("ch07", "corpus.txt").read_text(encoding="utf-8")


def test_pair_counts_hand_example():
    # "아아아" 한 단어 → ('▁','아') 1회, ('아','아') 2회
    counts = pair_counts(to_words("아아아"))
    assert counts[("▁", "아")] == 1 and counts[("아", "아")] == 2


def test_merge_symbols_left_to_right():
    assert merge_symbols(("a", "a", "a"), ("a", "a")) == ("aa", "a")


def test_first_merge_is_most_frequent():
    merges, _ = train_bpe(CORPUS, num_merges=1)
    (pair, freq), = merges
    assert freq == max(pair_counts(to_words(CORPUS)).values())


def test_more_merges_never_increase_tokens():
    merges, _ = train_bpe(CORPUS, num_merges=60)
    sample = "언어 모델은 토큰을 고른다."
    lengths = [len(encode(sample, merges[:n])) for n in range(0, 61, 5)]
    assert all(a >= b for a, b in zip(lengths, lengths[1:]))


@pytest.mark.parametrize("text", ["언어 모델은 토큰을 고른다.", "학습은 가중치를 고치는 일이다."])
def test_round_trip(text):
    merges, _ = train_bpe(CORPUS, num_merges=60)
    assert decode(encode(text, merges)) == text


def test_encoding_is_deterministic():
    merges, _ = train_bpe(CORPUS, num_merges=60)
    assert encode("모델은", merges) == encode("모델은", merges)


def test_vocab_size_is_base_plus_merges():
    merges, vocab = train_bpe(CORPUS, num_merges=60)
    base = {c for symbols in to_words(CORPUS) for c in symbols}
    assert len(vocab) == len(base) + len(merges)


def test_utf8_byte_lengths():
    assert [len(c.encode("utf-8")) for c in ("A", "토", "🙂")] == [1, 3, 4]


def test_qwen_vocab_equals_bytes_plus_merges():
    qwen_merges = pytest.importorskip("qwen_merges")
    try:
        merges = qwen_merges.load_merges(local_only=True)
    except Exception:
        pytest.skip("Qwen merges.txt가 아직 내려받아지지 않음(qwen_merges.py를 먼저 실행)")
    assert 256 + len(merges) == 151_643
