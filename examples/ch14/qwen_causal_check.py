# 제14장 핵심 실습: 실제 Qwen2.5-0.5B에서 미래 차단과 Padding Mask 확인
#
# 실행: python examples/ch14/qwen_causal_check.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.qwen import load_model, load_tokenizer  # noqa: E402

TEXT = "언어 모델은 다음 토큰을 예측한다"


def future_change_test(model, tok) -> tuple:
    """마지막 토큰만 바꿔 넣고, 앞 위치들과 마지막 위치의 Logits 차이를 잰다."""
    a = tok(TEXT, return_tensors="pt").input_ids
    b = a.clone()
    b[0, -1] = tok.encode(" 바나나")[0]
    with torch.no_grad():
        la, lb = model(a).logits, model(b).logits
    past = (la[0, :-1] - lb[0, :-1]).abs().max().item()
    last = (la[0, -1] - lb[0, -1]).abs().max().item()
    return past, last


def padding_test(model, tok) -> tuple:
    """긴 문장과 짧은 문장을 묶음으로 계산한 결과와, 짧은 문장을 단독으로 계산한 결과 비교."""
    short = "행렬곱"
    batch = tok([TEXT, short], return_tensors="pt", padding=True)
    with torch.no_grad():
        batched = model(**batch).logits
        alone = model(tok(short, return_tensors="pt").input_ids).logits
    n = alone.shape[1]
    return batch, (batched[1, :n] - alone[0]).abs().max().item()


def main() -> None:
    model, tok = load_model(), load_tokenizer()

    print("── 1. 마지막 토큰을 바꾸면 앞 위치의 예측이 변하는가 ──")
    past, last = future_change_test(model, tok)
    print(f"앞 위치 Logits 최대 차이: {past}")
    print(f"마지막 위치 Logits 최대 차이: {last:.2f}")

    print("\n── 2. Padding과 attention_mask ──")
    print(f"토크나이저 padding_side: {tok.padding_side},  PAD ID: {tok.pad_token_id}")
    batch, diff = padding_test(model, tok)
    print("짧은 문장의 input_ids:     ", batch.input_ids[1].tolist())
    print("짧은 문장의 attention_mask:", batch.attention_mask[1].tolist())
    print(f"묶음 계산과 단독 계산의 최대 차이(실제 토큰 위치): {diff:.1e}")


if __name__ == "__main__":
    main()
