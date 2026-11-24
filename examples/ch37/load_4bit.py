# 제37장 핵심 실습: bitsandbytes로 Qwen2.5-0.5B-Instruct를 NF4 4비트로 불러와 자료형·메모리·PPL을 확인한다
#
# 실행: python examples/ch37/load_4bit.py
#
# GPU(CC 6.0 이상)가 있으면 GPU에, 없으면 bitsandbytes CPU 경로로 불러온다.

import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "ch22"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # noqa: E402
from transformers.utils import logging  # noqa: E402

from common.fixed_data import load_blocks  # noqa: E402
from common.models import QWEN_ID, QWEN_REVISION  # noqa: E402
from common.qwen import load_tokenizer  # noqa: E402
from qwen_ppl import evaluate  # noqa: E402            (제22장)

logging.disable_progress_bar()
MiB = 2 ** 20


def hardware() -> dict:
    """bitsandbytes 기능별 하드웨어 조건(공식 문서: NF4 CC 6.0+, LLM.int8 CC 7.5+)"""
    if not torch.cuda.is_available():
        return {"device": "cpu", "name": "CPU", "cc": None, "nf4": True, "int8": True}
    major, minor = torch.cuda.get_device_capability(0)
    cc = major + minor / 10
    return {"device": "cuda", "name": torch.cuda.get_device_name(0), "cc": cc,
            "nf4": cc >= 6.0, "int8": cc >= 7.5}


def nf4_config() -> BitsAndBytesConfig:
    """제36장과 같은 NF4 + 이중 양자화, 계산은 FP32(GT 1030 기준)"""
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)


def load_nf4(device: str):
    return AutoModelForCausalLM.from_pretrained(
        QWEN_ID, revision=QWEN_REVISION, quantization_config=nf4_config(),
        dtype=torch.float32, device_map=device).eval()       # dtype: 양자화하지 않는 계층의 자료형


def memory_parts(model) -> dict:
    """파라미터·버퍼 바이트를 '4비트 가중치'와 '그 밖'으로 나눈다(4비트 메타데이터는 따로)."""
    parts = Counter()
    for name, p in model.named_parameters():
        key = "4비트 가중치(번호)" if p.dtype == torch.uint8 else (
            "Embedding(= LM Head)" if "embed_tokens" in name else "편향·RMSNorm")
        parts[key] += p.numel() * p.element_size()
    meta = 0
    for mod in model.modules():
        qs = getattr(getattr(mod, "weight", None), "quant_state", None)
        if qs is not None:
            meta += qs.absmax.numel() * qs.absmax.element_size()
            if qs.state2 is not None:
                meta += qs.state2.absmax.numel() * qs.state2.absmax.element_size()
    parts["4비트 메타데이터(absmax)"] = meta
    return dict(parts)


def main() -> None:
    hw = hardware()
    print("── 1. 하드웨어 조건 ──")
    print(f"  장치 {hw['name']}, Compute Capability {hw['cc'] or '해당 없음'}")
    print(f"  NF4 4비트: {'가능' if hw['nf4'] else '불가'}, LLM.int8: {'가능' if hw['int8'] else '불가'}"
          f"{' (CPU 경로)' if hw['device'] == 'cpu' else ''}")

    print("\n── 2. BitsAndBytesConfig(제36장과의 대응) ──")
    c = nf4_config()
    for k, meaning in (("bnb_4bit_quant_type", "Codebook(제36장 NF4)"),
                       ("bnb_4bit_use_double_quant", "absmax 이중 양자화"),
                       ("bnb_4bit_compute_dtype", "역양자화 후 계산 자료형")):
        print(f"  {k:<27}{str(getattr(c, k)):<16}{meaning}")

    print("\n── 3. 4비트 로딩: 계층 종류와 자료형 ──")
    tok = load_tokenizer()
    model = load_nf4(hw["device"])
    kinds = Counter(type(m).__name__ for m in model.modules() if not list(m.children()))
    print(f"  말단 계층 {dict(kinds)}")
    q = model.model.layers[0].self_attn.q_proj
    print(f"  q_proj: {type(q).__name__}, 저장 {q.weight.dtype} {tuple(q.weight.shape)} "
          f"(896 × 896 ÷ 2 = {896 * 896 // 2:,}바이트), 계산 {q.compute_dtype}")
    qs = q.weight.quant_state
    print(f"  블록 크기 {qs.blocksize}, 이중 양자화 블록 {qs.state2.blocksize}, Codebook 값 {qs.code.numel()}개")
    print(f"  Embedding {model.model.embed_tokens.weight.dtype}, LM Head 공유 "
          f"{model.lm_head.weight is model.model.embed_tokens.weight}")

    print("\n── 4. 메모리(파라미터 기준) ──")
    parts = memory_parts(model)
    for k, v in parts.items():
        print(f"  {k:<26}{v / MiB:8.1f}MiB")
    total = sum(parts.values())
    print(f"  {'합계':<26}{total / MiB:8.1f}MiB  (FP32 전체 {494_032_768 * 4 / MiB:,.0f}MiB의 "
          f"{total / (494_032_768 * 4):.0%})")
    print(f"  get_memory_footprint() {model.get_memory_footprint() / MiB:.1f}MiB")
    if hw["device"] == "cuda":
        print(f"  GPU 최대 할당 {torch.cuda.max_memory_allocated() / MiB:.0f}MiB")

    print("\n── 5. 고정 평가 데이터 PPL ──")
    r = evaluate(model, load_blocks(tok, "eval"), hw["device"])
    print(f"  bitsandbytes NF4 + 이중 양자화  PPL {math.exp(r['loss']):.2f}")
    print("  (참고: FP32 21.45, 제36장 직접 구현 NF4 + 이중 양자화 27.87)")


if __name__ == "__main__":
    main()
