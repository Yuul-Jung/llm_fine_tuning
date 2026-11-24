# 제09장 실습 2: 실제 Qwen2.5 토크나이저의 병합 규칙 확인
#
# 실행: python examples/ch09/qwen_merges.py
#
# Byte-level BPE의 어휘 = 기본 바이트 256개 + 학습된 병합 규칙 수

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.models import qwen_file  # noqa: E402


def load_merges(local_only: bool = False) -> list:
    path = qwen_file("merges.txt", local_only=local_only)
    with open(path, encoding="utf-8") as f:
        return [line.split() for line in f.read().split("\n") if line.strip()]


def main() -> None:
    merges = load_merges()
    print(f"병합 규칙 수: {len(merges):,}")
    print(f"기본 바이트 256 + 병합 {len(merges):,} = {256 + len(merges):,}  (일반 토큰 수와 비교)")
    print("\n처음 6개 규칙(Ġ는 공백 바이트를 표시하는 기호):")
    for i, (a, b) in enumerate(merges[:6], 1):
        print(f"  {i}. {a!r} + {b!r} → {a + b!r}")


if __name__ == "__main__":
    main()
