# 제20장 실습 1: Auto 클래스로 불러온 실제 Qwen2.5-0.5B의 설정·계층 이름·Shape·생성 설정 살펴보기
#
# 실행: python examples/ch20/inspect_qwen.py

import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.models import qwen_file  # noqa: E402
from common.qwen import load_config, load_model, load_tokenizer  # noqa: E402


def layer_table(model) -> "OrderedDict[str, tuple]":
    """계층 번호를 N으로 바꿔 24개 계층을 한 줄로 묶는다."""
    table = OrderedDict()
    for name, p in model.named_parameters():
        key = re.sub(r"layers\.\d+\.", "layers.N.", name)
        table.setdefault(key, (tuple(p.shape), 0))
        table[key] = (table[key][0], table[key][1] + 1)
    return table


def main() -> None:
    config, tok, model = load_config(), load_tokenizer(), load_model()

    print("── 1. Auto 클래스가 고른 실제 클래스 ──")
    print(f"  설정 {type(config).__name__} / 토크나이저 {type(tok).__name__} / 모델 {type(model).__name__}")
    print(f"  config.json의 model_type: {config.model_type}, architectures: {config.architectures}")

    print("\n── 2. 계층 이름과 Shape(N = 0~23) ──")
    for key, (shape, n) in layer_table(model).items():
        print(f"  {key:<48}{str(shape):<14}×{n}")
    print(f"  lm_head.weight는 embed_tokens.weight와 같은 텐서: "
          f"{model.lm_head.weight is model.model.embed_tokens.weight}")

    print("\n── 3. generation_config.json(생성 기본값) ──")
    gen = json.loads(Path(qwen_file("generation_config.json")).read_text())
    for k in ("do_sample", "temperature", "top_k", "top_p", "repetition_penalty", "eos_token_id"):
        print(f"  {k:<20}{gen[k]}")


if __name__ == "__main__":
    main()
