# 본문 코드 박스와 examples/ 실습 소스의 일치 여부 검사
#
# 실행: python tools/check_code_boxes.py
# 기준: content/chNN.md에서 "파일명: `경로`" 바로 아래 코드 박스가 실제 파일과 한 글자도 다르지 않아야 한다.

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"파일명: `([^`]+)`\n\n```[a-z]*\n(.*?)```", re.S)


def main() -> int:
    failures = 0
    for md in sorted((ROOT / "content").glob("ch*.md")):
        for name, code in PATTERN.findall(md.read_text(encoding="utf-8")):
            source = ROOT / name
            if not source.is_file():
                print(f"[없음] {md.name}: {name}")
                failures += 1
            elif source.read_text(encoding="utf-8") != code:
                print(f"[불일치] {md.name}: {name}")
                failures += 1
            else:
                print(f"[일치] {md.name}: {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
