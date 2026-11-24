# 제08장 핵심 실습: Qwen2.5 토크나이저로 한글·영문 문장을 토큰화하고 되돌리기
#
# 실행: python examples/ch08/tokenize_compare.py
# 준비: 처음 실행할 때 토크나이저 파일(수 MB)을 내려받는다.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer  # noqa: E402

from common.models import QWEN_ID, QWEN_REVISION  # noqa: E402

SENTENCES = [
    "언어 모델은 다음 토큰을 예측한다.",
    "A language model predicts the next token.",
]


def load_tokenizer(local_only: bool = False):
    return AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION,
                                         local_files_only=local_only)


def split_three_ways(text: str, tok) -> dict:
    """같은 문장을 문자·단어·서브워드 단위로 나눈 개수."""
    return {
        "문자": len(text),
        "단어(공백 기준)": len(text.split()),
        "서브워드(Qwen)": len(tok.encode(text)),
    }


def main() -> None:
    tok = load_tokenizer()

    print("── 1. 어휘 집합과 특수 토큰 ──")
    print(f"일반 토큰 수: {tok.vocab_size:,},  특수 토큰 포함 전체: {len(tok):,}")
    print(f"BOS: {tok.bos_token},  EOS: {tok.eos_token} ({tok.eos_token_id}),  "
          f"PAD: {tok.pad_token} ({tok.pad_token_id})")
    for name in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        print(f"  {name:<15} → ID {tok.convert_tokens_to_ids(name)}")

    print("\n── 2. 문장 토큰화와 되돌리기 ──")
    for text in SENTENCES:
        ids = tok.encode(text)
        pieces = [tok.decode([i]) for i in ids]
        print(f"\n문장: {text}")
        print(f"  Token ID ({len(ids)}개): {ids}")
        print(f"  토큰별 글자: {pieces}")
        print(f"  디코딩 결과가 원문과 같은가: {tok.decode(ids) == text}")

    print("\n── 3. 문자·단어·서브워드 개수 비교 ──")
    for text in SENTENCES:
        counts = split_three_ways(text, tok)
        print(f"  {text[:20]:<22}" + "  ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
