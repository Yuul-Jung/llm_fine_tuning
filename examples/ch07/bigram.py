# 제07장 핵심 실습: 글자 단위 Bigram 언어 모델로 자기회귀 생성하기
#
# 실행: python examples/ch07/bigram.py
#
# Bigram: 바로 앞 글자 하나만 보고 다음 글자의 확률을 정하는 가장 작은 언어 모델

from pathlib import Path

import torch

BOS, EOS = "^", "$"      # 문장 시작·끝 표시(특수 토큰)


class BigramLM:
    def __init__(self, lines: list, alpha: float = 0.01):
        chars = sorted({c for line in lines for c in line})
        self.vocab = [BOS, EOS] + chars                   # 어휘 집합
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        V = len(self.vocab)

        # 1) 빈도표: counts[앞 글자, 다음 글자] = 함께 나온 횟수
        self.counts = torch.zeros(V, V)
        for line in lines:
            seq = [BOS] + list(line) + [EOS]
            for a, b in zip(seq, seq[1:]):
                self.counts[self.stoi[a], self.stoi[b]] += 1

        # 2) Logits = log(빈도 + α) → Softmax → 확률표
        #    α는 한 번도 나오지 않은 조합의 확률이 정확히 0이 되지 않게 하는 작은 값
        self.logits = torch.log(self.counts + alpha)
        self.probs = torch.softmax(self.logits, dim=-1)   # 각 행의 합 = 1

    def next_probs(self, prev: str) -> torch.Tensor:
        """앞 글자가 prev일 때 다음 글자의 확률분포 P(다음 | prev)."""
        return self.probs[self.stoi[prev]]

    def sequence_log_prob(self, text: str) -> float:
        """문장 확률 = 조건부 확률의 곱 → 로그를 취해 합으로 계산한다."""
        seq = [BOS] + list(text) + [EOS]
        return sum(torch.log(self.next_probs(a)[self.stoi[b]]).item()
                   for a, b in zip(seq, seq[1:]))

    def generate(self, prefix: str = "", max_new: int = 40, greedy: bool = True,
                 seed: int = 0) -> str:
        """자기회귀 생성: 고른 글자를 문맥 끝에 붙이고 다시 다음 글자를 고른다."""
        generator = torch.Generator().manual_seed(seed)
        out = prefix
        last = prefix[-1] if prefix else BOS
        for _ in range(max_new):
            p = self.next_probs(last)
            if greedy:
                idx = int(torch.argmax(p))                         # 가장 높은 확률
            else:
                idx = int(torch.multinomial(p, 1, generator=generator))  # 확률대로 뽑기
            ch = self.vocab[idx]
            if ch == EOS:
                break
            out += ch
            last = ch
        return out


def load_corpus() -> list:
    text = Path(__file__).with_name("corpus.txt").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def main() -> None:
    lm = BigramLM(load_corpus())
    print(f"어휘 크기: {len(lm.vocab)}개 (특수 토큰 2 + 글자 {len(lm.vocab) - 2})")
    print(f"확률표 모양: {tuple(lm.probs.shape)}, 행 합의 최솟값: {lm.probs.sum(dim=-1).min():.6f}")

    print("\n── 1. '모' 다음에 올 글자 상위 3개 ──")
    p = lm.next_probs("모")
    top = torch.topk(p, 3)
    for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  P({lm.vocab[idx]!r} | '모') = {prob:.3f}  (빈도 {int(lm.counts[lm.stoi['모'], idx])}회)")

    print("\n── 2. 문장 확률 = 조건부 확률의 곱 ──")
    for s in ("언어 모델", "모델 언어"):
        print(f"  log P({s!r}) = {lm.sequence_log_prob(s):.2f}")

    print("\n── 3. Greedy Decoding ──")
    for prefix in ("언", "학", "확"):
        print(f"  {prefix!r} → {lm.generate(prefix)!r}")

    print("\n── 4. 확률대로 뽑기(Sampling, seed 고정) ──")
    for seed in (0, 1, 2):
        print(f"  seed {seed}: {lm.generate('언', greedy=False, seed=seed)!r}")


if __name__ == "__main__":
    main()
