# 제07장 검증 테스트
#
# 실행: python -m pytest examples/ch07 -q

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bigram import BOS, EOS, BigramLM, load_corpus  # noqa: E402


def make_model():
    return BigramLM(load_corpus())


def test_rows_sum_to_one():
    lm = make_model()
    torch.testing.assert_close(lm.probs.sum(dim=-1), torch.ones(len(lm.vocab)))


def test_bigram_count_total():
    # 문장마다 (글자 수 + 1)개의 쌍: [BOS, c1], [c1, c2], ..., [cn, EOS]
    lines = load_corpus()
    assert int(make_model().counts.sum()) == sum(len(line) + 1 for line in lines)


def test_probability_formula():
    # Softmax(log(count + α)) = (count + α) / Σ(count + α)
    lm = make_model()
    row = lm.counts[lm.stoi["모"]] + 0.01
    torch.testing.assert_close(lm.next_probs("모"), row / row.sum())


def test_most_likely_after_mo():
    lm = make_model()
    assert lm.vocab[int(torch.argmax(lm.next_probs("모")))] == "델"


def test_word_order_probability():
    lm = make_model()
    assert lm.sequence_log_prob("언어 모델") > lm.sequence_log_prob("모델 언어")


def test_greedy_is_reproducible():
    lm = make_model()
    assert lm.generate("언") == lm.generate("언")


def test_sampling_reproducible_with_seed():
    lm = make_model()
    assert lm.generate("언", greedy=False, seed=1) == lm.generate("언", greedy=False, seed=1)


def test_generation_stops_at_eos():
    # 'EOS' 확률이 가장 높은 글자('.' 다음)에서 생성이 끝나야 한다.
    lm = make_model()
    assert lm.vocab[int(torch.argmax(lm.next_probs(".")))] == EOS
    assert lm.generate(".") == "."


def test_special_tokens_in_vocab():
    lm = make_model()
    assert lm.vocab[:2] == [BOS, EOS]
