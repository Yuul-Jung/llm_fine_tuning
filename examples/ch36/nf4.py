# 제36장 핵심 실습: NF4(NormalFloat 4비트)와 블록 단위·이중 양자화를 직접 구현해 INT4와 비교한다
#
# 실행: python examples/ch36/nf4.py
#
# NF4: 정규분포의 분위수로 만든 16개 값(Codebook) 중 가장 가까운 것을 고른다.
#      블록(64개)마다 절댓값 최대(absmax)로 [−1, 1]에 맞춘 뒤 Codebook에서 고르고, 번호(4비트)를 저장한다.

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "ch22", HERE.parent / "ch33", HERE.parent / "ch34", HERE.parent / "ch35"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from common.fixed_data import load_blocks  # noqa: E402
from common.qwen import load_model, load_tokenizer  # noqa: E402
from int4_pack import dequant_int4, quant_int4  # noqa: E402            (제34장)
from int8_quant import NAME  # noqa: E402                               (제33장)
from qwen_ppl import evaluate  # noqa: E402                             (제22장)
from quant_error import fake_quantize_model  # noqa: E402               (제35장)

BLOCK, BLOCK2 = 64, 256


def nf4_codebook(offset: float = 0.9677083) -> torch.Tensor:
    """정규분포 분위수로 16개 값을 만든다: 양수 8개 + 음수 7개 + 0, 최댓값으로 나눠 [−1, 1]"""
    normal = torch.distributions.Normal(0.0, 1.0)
    pos = normal.icdf(torch.linspace(offset, 0.5, 9, dtype=torch.float64)[:-1])     # 양수 8개
    neg = -normal.icdf(torch.linspace(offset, 0.5, 8, dtype=torch.float64)[:-1])    # 음수 7개
    values = torch.cat([pos, neg, torch.zeros(1, dtype=torch.float64)]).sort().values
    return (values / values.max()).float()


CODE = nf4_codebook()


def nf4_quant(w: torch.Tensor, block: int = BLOCK) -> tuple:
    """(4비트 번호 0~15, 블록별 absmax) 반환"""
    blocks = w.reshape(-1, block)
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    idx = (blocks / absmax).unsqueeze(-1).sub(CODE).abs().argmin(dim=-1)   # 가장 가까운 Codebook 값
    return idx.to(torch.uint8), absmax


def nf4_dequant(idx: torch.Tensor, absmax: torch.Tensor, shape) -> torch.Tensor:
    return (CODE[idx.long()] * absmax).reshape(shape)


def double_quant(absmax: torch.Tensor, block2: int = BLOCK2) -> tuple:
    """absmax(FP32)를 평균을 뺀 뒤 256개씩 INT8 대칭 양자화한다(QLoRA의 이중 양자화를 단순화)."""
    a = absmax.flatten()
    mean = a.mean()
    c = (a - mean).reshape(-1, block2)
    s2 = c.abs().amax(dim=1, keepdim=True) / 127
    return torch.round(c / s2).clamp(-127, 127).to(torch.int8), s2, mean


def double_dequant(q2, s2, mean, shape) -> torch.Tensor:
    return (q2.float() * s2).flatten().add(mean).reshape(shape)


def nf4_fake(w: torch.Tensor, dq: bool = False) -> torch.Tensor:
    idx, absmax = nf4_quant(w)
    if dq:
        absmax = double_dequant(*double_quant(absmax), absmax.shape)
    return nf4_dequant(idx, absmax, w.shape)


def avg_bits(block: int = BLOCK, dq: bool = False, block2: int = BLOCK2) -> float:
    """가중치 하나당 평균 비트 = 4 + Scale 비용"""
    return 4 + (8 / block + 32 / (block * block2) if dq else 32 / block)


def main() -> None:
    print("── 1. NF4 Codebook(정규분포 분위수로 만든 16개 값) ──")
    print("  " + ", ".join(f"{v:+.4f}" for v in CODE.tolist()))
    print("  균일 INT4(÷7):  " + ", ".join(f"{v / 7:+.3f}" for v in range(-7, 8)))
    try:
        import bitsandbytes.functional as bF
        ref = bF.get_4bit_type("nf4", device="cpu")
        print(f"  bitsandbytes NF4 표와 최대 차이: {(ref - CODE).abs().max().item():.1e}")
    except Exception as e:                                         # 설치되지 않은 환경
        print(f"  bitsandbytes 비교 생략({type(e).__name__})")

    w = dict(load_model().named_parameters())[NAME].detach()
    print(f"\n── 2. {NAME}: 블록 64, 같은 4비트 비교 ──")
    blocks = (w.reshape(-1, BLOCK) / w.reshape(-1, BLOCK).abs().amax(dim=1, keepdim=True)).flatten()
    within = (blocks.abs() < 0.5).float().mean().item()
    n_int4 = sum(abs(v / 7) < 0.5 for v in range(-7, 8))
    n_nf4 = int((CODE.abs() < 0.5).sum())
    print(f"  블록 최댓값으로 나눈 값의 {within:.0%}가 −0.5 ~ 0.5에 있다"
          f"(이 구간의 단계: 균일 INT4 {n_int4}개, NF4 {n_nf4}개)")
    rows = {"INT4 Group 64": dequant_int4(*quant_int4(w, 64), 64),
            "NF4 블록 64": nf4_fake(w), "NF4 블록 64 + 이중 양자화": nf4_fake(w, dq=True)}
    for label, back in rows.items():
        e = back - w
        print(f"  {label:<26}상대 오차 {(e.norm() / w.norm()).item():.4f}, MSE {e.pow(2).mean().item():.2e}, "
              f"최대 절대오차 {e.abs().max().item():.4f}")
    try:
        import bitsandbytes.functional as bF
        q, st = bF.quantize_4bit(w.contiguous(), blocksize=BLOCK, quant_type="nf4", compress_statistics=False)
        theirs = bF.dequantize_4bit(q, st)
        print(f"  bitsandbytes NF4(CPU)와 직접 구현의 최대 차이 {(theirs - rows['NF4 블록 64']).abs().max().item():.1e}")
    except Exception as e:
        print(f"  bitsandbytes 비교 생략({type(e).__name__})")

    print("\n── 3. 가중치 하나당 평균 비트 ──")
    for label, b in (("INT4 Group 64(FP32 Scale)", avg_bits()), ("NF4 블록 64(FP32 absmax)", avg_bits()),
                     ("NF4 블록 64 + 이중 양자화", avg_bits(dq=True))):
        print(f"  {label:<28}{b:.3f}비트")

    print("\n── 4. 모델 전체 PPL(제35장과 같은 조건) ──")
    tok = load_tokenizer()
    eval_blocks = load_blocks(tok, "eval")
    for label, f in (("INT4 Group 64", lambda x: dequant_int4(*quant_int4(x, 64), 64)),
                     ("NF4 블록 64", nf4_fake), ("NF4 + 이중 양자화", lambda x: nf4_fake(x, dq=True))):
        m = load_model()
        fake_quantize_model(m, f)
        print(f"  {label:<18}PPL {math.exp(evaluate(m, eval_blocks)['loss']):.2f}")
        del m


if __name__ == "__main__":
    main()
