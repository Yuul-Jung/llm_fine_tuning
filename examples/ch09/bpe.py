# 제09장 핵심 실습: 교육용 BPE 토크나이저 직접 구현
#
# 실행: python examples/ch09/bpe.py
#
# BPE(Byte Pair Encoding): 가장 자주 붙어 나오는 두 토큰을 하나로 합치는 일을 반복해 어휘를 만든다.
# 단어 앞에 '▁'를 붙여 공백 위치를 기억한다(디코딩 때 공백으로 되돌림).

from collections import Counter
from pathlib import Path

SPACE = "▁"


def to_words(text: str) -> Counter:
    """문장을 공백으로 나누고, 단어마다 글자 튜플로 바꿔 등장 횟수를 센다."""
    words = Counter()
    for word in text.split():
        words[tuple(SPACE + word)] += 1
    return words


def pair_counts(words: Counter) -> Counter:
    """모든 단어 안에서 이웃한 두 토큰 쌍의 빈도를 센다(단어 빈도만큼 가중)."""
    counts = Counter()
    for symbols, freq in words.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] += freq
    return counts


def merge_symbols(symbols: tuple, pair: tuple) -> tuple:
    """토큰열 안의 pair를 모두 하나의 토큰으로 합친다(왼쪽부터)."""
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(text: str, num_merges: int) -> tuple:
    """병합 규칙(merges)을 순서대로 학습한다. 동점이면 먼저 나온 쌍을 고른다."""
    words = to_words(text)
    base_vocab = sorted({c for symbols in words for c in symbols})
    merges = []
    for _ in range(num_merges):
        counts = pair_counts(words)
        if not counts:
            break
        best, freq = counts.most_common(1)[0]
        merges.append((best, freq))
        words = Counter({merge_symbols(s, best): f for s, f in words.items()})
    vocab = base_vocab + ["".join(pair) for pair, _ in merges]
    return merges, vocab


def encode(text: str, merges: list) -> list:
    """새 문장에 학습된 병합 규칙을 학습 순서대로 적용한다."""
    tokens = []
    for word in text.split():
        symbols = tuple(SPACE + word)
        for pair, _ in merges:
            symbols = merge_symbols(symbols, pair)
        tokens.extend(symbols)
    return tokens


def decode(tokens: list) -> str:
    return "".join(tokens).replace(SPACE, " ").strip()


def main() -> None:
    corpus = Path(__file__).parents[1].joinpath("ch07", "corpus.txt").read_text(encoding="utf-8")
    sample = "언어 모델은 토큰을 고른다."

    print("── 1. 병합 과정(처음 10번) ──")
    merges, vocab = train_bpe(corpus, num_merges=60)
    for step, (pair, freq) in enumerate(merges[:10], 1):
        print(f"{step:>2}. {pair[0]!r} + {pair[1]!r} → {pair[0] + pair[1]!r}  (빈도 {freq})")

    print("\n── 2. 병합 횟수에 따른 토큰 수 ──")
    for n in (0, 10, 30, 60):
        tokens = encode(sample, merges[:n])
        print(f"병합 {n:>2}번: {len(tokens):>2}개  {tokens}")

    print("\n── 3. 왕복 복원 ──")
    tokens = encode(sample, merges)
    print(f"{sample!r} → {tokens} → {decode(tokens)!r}")

    print("\n── 4. 처음 보는 단어 ──")
    unseen = "모델링"
    tokens = encode(unseen, merges)
    missing = [tok for tok in tokens if tok not in vocab]
    print(f"{unseen!r} → {tokens}")
    print(f"  '▁모델'은 학습한 조각으로 표현되지만, 어휘에 없는 토큰이 남는다: {missing}")

    print("\n── 5. Byte-level: 모든 글자는 바이트로 표현할 수 있다 ──")
    for ch in ("A", "토", "🙂"):
        data = list(ch.encode("utf-8"))
        print(f"{ch!r}: UTF-8 {len(data)}바이트 {data}")
    print(f"어휘 크기: 기본 글자 {len(vocab) - len(merges)}개 + 병합 {len(merges)}개 = {len(vocab)}개")


if __name__ == "__main__":
    main()
