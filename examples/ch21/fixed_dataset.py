# 제21장 실습 2: 제3·4부 비교 실험의 기준이 되는 고정 토큰 데이터 만들기와 지문 확인
#
# 실행: python examples/ch21/fixed_dataset.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.fixed_data import SEQ_LEN, fingerprint, load_blocks, read_lines, token_stream  # noqa: E402
from common.qwen import load_tokenizer  # noqa: E402


def main() -> None:
    tok = load_tokenizer()
    print(f"── 1. 문장 파일 → 토큰 → 길이 {SEQ_LEN} 블록 ──")
    for split in ("train", "eval"):
        lines = read_lines(split)
        chars = sum(len(s) for s in lines)
        ids = token_stream(tok, split)
        blocks = load_blocks(tok, split)
        print(f"  {split:<6}문장 {len(lines):>3}개, 글자 {chars:>5,}, 토큰 {len(ids):>5,} "
              f"(글자당 {len(ids) / chars:.2f}) → 블록 {tuple(blocks.shape)}, "
              f"버린 꼬리 {len(ids) - blocks.numel()}")

    print("\n── 2. 첫 블록의 앞부분 ──")
    first = load_blocks(tok, "train")[0]
    print(f"  토큰 {first[:12].tolist()}")
    print(f"  복원 {tok.decode(first[:40])!r}")

    print("\n── 3. 지문(실험 비교의 기준값) ──")
    for split in ("train", "eval"):
        a, b = load_blocks(tok, split), load_blocks(tok, split)
        print(f"  {split:<6}{fingerprint(a)}  다시 만들어도 같음: {fingerprint(a) == fingerprint(b)}")


if __name__ == "__main__":
    main()
