# 제55장 핵심 실습: 병합 모델을 Q8_0·Q4_0 GGUF로 양자화하고, 크기·로딩·PPL·정답률을 F32·F16과 비교한다
#
# 실행: python examples/ch55/quant_compare.py   (CPU 약 30분, 결과는 quant.json에 저장되어 이어서 실행 가능)
#
# 품질(PPL·핵심어·형식)은 GGUF를 Transformers로 다시 읽어(FP32로 풀어서) 잰다.
# llama.cpp에서의 초당 토큰 수와 실제 메모리, GT 1030 실측은 교재 검증 환경에서 하지 않았다(미검증).

import gc
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent, HERE.parent / "ch25", HERE.parent / "ch43", HERE.parent / "ch47", HERE.parent / "ch54",
          HERE.parent / "ch58"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gguf import GGMLQuantizationType, quants  # noqa: E402
from transformers.utils import logging  # noqa: E402

from baseline import measure  # noqa: E402                    (제47장)
from criteria import judge, load_criteria  # noqa: E402        (제43장)
from ondevice import DEVICES, fits, kv_mib  # noqa: E402       (제58장)
from to_gguf import convert  # noqa: E402                      (제54장)

logging.disable_progress_bar()
logging.set_verbosity_error()
MERGED = ROOT / "models" / "fpga_merged_v1"
GGUF = ROOT / "models" / "gguf"
OUT = HERE / "quant.json"
FORMATS = ["F32", "F16", "Q8_0", "Q4_0"]
MiB = 2 ** 20


def q8_0_by_hand(x: np.ndarray) -> np.ndarray:
    """32개 블록마다 d = max|x| / 127, q = round(x / d)(0.5는 0에서 먼 쪽으로) → 복원 q × fp16(d) (제33장의 대칭 INT8)"""
    b = x.reshape(-1, 32)
    d = np.abs(b).max(axis=1, keepdims=True) / 127
    q = np.trunc(b / d + np.copysign(0.5, b / d))
    return (q * d.astype(np.float16).astype(np.float32)).reshape(x.shape)


def q4_0_by_hand(x: np.ndarray) -> np.ndarray:
    """32개 블록마다 절댓값이 가장 큰 원소 m으로 d = m / -8, q = ⌊x / d + 8.5⌋ ∈ [0, 15] → 복원 (q - 8) × fp16(d)"""
    b = x.reshape(-1, 32)
    m = b[np.arange(len(b)), np.abs(b).argmax(axis=1)][:, None]
    d = m / -8
    q = np.clip(np.trunc(b / d + 8.5), 0, 15)
    return ((q - 8) * d.astype(np.float16).astype(np.float32)).reshape(x.shape)


def load_gguf(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(GGUF, gguf_file=f"{name}.gguf", dtype=torch.float32).eval()
    model.generation_config.repetition_penalty = 1.1          # 원본 생성 설정을 옮긴다(제54장)
    tok = AutoTokenizer.from_pretrained(MERGED)
    return tok, model, time.time() - t0


def main() -> None:
    from common.fixed_data import load_blocks
    from trainer import eval_loss
    rec = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    print("── 1. 블록 양자화 구조: 직접 구현 ↔ gguf 라이브러리 ──")
    x = np.random.default_rng(0).standard_normal((8, 896)).astype(np.float32)
    for q, fn, size in (("Q8_0", q8_0_by_hand, 34), ("Q4_0", q4_0_by_hand, 18)):
        lib = quants.dequantize(quants.quantize(x, GGMLQuantizationType[q]), GGMLQuantizationType[q])
        print(f"  {q}: 블록 32개 = {size}바이트({size * 8 / 32}비트/값), 직접 구현과 최대 차이 "
              f"{np.abs(fn(x) - lib).max():.1e}, 복원 오차(평균 제곱) {np.mean((lib - x) ** 2):.2e}")

    print("\n── 2. 형식별 GGUF 만들기 ──")
    for f in FORMATS:
        path = GGUF / f"fpga-{f.lower()}.gguf"
        if not path.exists():
            t0 = time.time()
            convert(MERGED, path, f)
            print(f"  {f}: 변환 {time.time() - t0:.0f}초")
        rec.setdefault(f, {})["size_mib"] = path.stat().st_size / MiB
    print("  " + ", ".join(f"{f} {rec[f]['size_mib']:.0f}MiB" for f in FORMATS))

    print("\n── 3. 형식별 품질(validation, 제43장 기준) ──")
    c = load_criteria()
    base = json.loads((ROOT / "examples/ch47/baseline.json").read_text(encoding="utf-8"))
    for f in FORMATS:
        if "domain_ppl" in rec[f]:
            continue
        tok, model, load_s = load_gguf(f"fpga-{f.lower()}")
        m = measure(tok, model)
        hits, fmts = list(m["keyword_hit"].values()), list(m["format_ok"].values())
        v = judge(c, domain_ratio=m["domain_ppl"] / base["domain_ppl"],
                  general_ratio=m["general_ppl"] / base["general_ppl"], hits=hits, fmts=fmts)
        rec[f].update(load_s=load_s, domain_ppl=m["domain_ppl"], general_ppl=m["general_ppl"],
                      keyword=sum(hits) / len(hits), format=sum(fmts) / len(fmts), passed=sum(v.values()),
                      answers=m["answers"])
        OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        del model
        gc.collect()
    print("  형식   크기MiB  도메인PPL  일반PPL  핵심어  형식   합격  F32와 같은 답")
    for f in FORMATS:
        r = rec[f]
        same = sum(r["answers"][i] == rec["F32"]["answers"][i] for i in r["answers"])
        print(f"  {f:<6}{r['size_mib']:>7.0f}{r['domain_ppl']:>10.2f}{r['general_ppl']:>9.2f}"
              f"{r['keyword']:>7.2f}{r['format']:>6.2f}   {r['passed']}/4   {same}/6")

    print("\n── 4. GT 1030 탑재 판정(가중치 + FP16 KV Cache 2048토큰 + 연산 버퍼 가정 300MiB) ──")
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MERGED)
    for f in FORMATS:
        need = rec[f]["size_mib"] + kv_mib(cfg, 2048, 2) + 300
        print(f"  {f:<6}{need:>7.0f}MiB → 2GB의 75% 이내 {fits(need, DEVICES['GT 1030'])}")
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
