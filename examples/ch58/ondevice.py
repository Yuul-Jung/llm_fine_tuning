# 제58장 핵심 실습: 장치별 메모리 예산(가중치 + KV Cache + 여유)을 계산하고, CPU에서 속도·메모리를 측정한다
#
# 실행: python examples/ch58/ondevice.py   (제52장의 병합 모델 필요, CPU 약 2분)
#
# GGUF 형식의 크기는 블록 구조로 계산한 값이다(Q8_0: 32개당 34바이트, Q4_0: 32개당 18바이트).
# llama.cpp·GT 1030·Raspberry Pi 실측은 교재 검증 환경에서 하지 않았다(미검증).

import json
import platform
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent, HERE.parent / "ch20"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from common.device import select_device  # noqa: E402
from sampling_kv import kv_bytes_per_token  # noqa: E402        (제20장)

MERGED = ROOT / "models" / "fpga_merged_v1"
OUT = HERE / "measure.json"
MiB = 2 ** 20
BITS = {"FP32": 32, "FP16": 16, "Q8_0": 34 * 8 / 32, "Q4_0": 18 * 8 / 32}     # 가중치 하나당 평균 비트
DEVICES = {"GT 1030": 2048, "Raspberry Pi 4GB": 4096, "교재 CPU 8GB": 8192}  # 쓸 수 있는 메모리 MiB
RESERVE = 0.25                                                                 # 운영체제·연산 버퍼 몫


def weight_mib(n_params: int, fmt: str) -> float:
    return n_params * BITS[fmt] / 8 / MiB


def kv_mib(cfg, tokens: int, bytes_per: int) -> float:
    return kv_bytes_per_token(cfg, bytes_per) * tokens / MiB


def fits(total_mib: float, device_mib: int) -> bool:
    return total_mib <= device_mib * (1 - RESERVE)


def rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024        # 리눅스: KiB 단위


def main() -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(MERGED)
    n = 494_032_768                                            # 제52장 병합 모델 파라미터 수
    print("── 1. KV Cache: 토큰 하나당과 문맥 길이별(Qwen2.5-0.5B) ──")
    print(f"  토큰당 FP32 {kv_bytes_per_token(cfg, 4):,}B, FP16 {kv_bytes_per_token(cfg, 2):,}B "
          f"(2 × {cfg.num_hidden_layers}층 × KV 헤드 {cfg.num_key_value_heads} × 64 × 자료형)")
    for t in (512, 2048, 8192, 32768):
        print(f"  {t:>6}토큰  FP32 {kv_mib(cfg, t, 4):7.1f}MiB  FP16 {kv_mib(cfg, t, 2):7.1f}MiB")

    print("\n── 2. 장치별 탑재 판정(가중치 + FP16 KV Cache, 가용의 75% 이내) ──")
    print(f"  가중치: " + ", ".join(f"{f} {weight_mib(n, f):.0f}MiB" for f in BITS))
    table = {}
    for dev, mem in DEVICES.items():
        row = {}
        for fmt in BITS:
            for t in (2048, 32768):
                row[f"{fmt}@{t}"] = fits(weight_mib(n, fmt) + kv_mib(cfg, t, 2), mem)
        table[dev] = row
        ok = [k for k, v in row.items() if v]
        print(f"  {dev:<17}가능: {', '.join(ok) if ok else '없음'}")

    print("\n── 3. CPU 실측(병합 모델 FP32, Hugging Face, Greedy 64토큰) ──")
    device, _ = select_device()
    tok = AutoTokenizer.from_pretrained(MERGED)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MERGED, dtype=torch.float32).to(device).eval()
    load_s = time.time() - t0
    rss_loaded = rss_mib()
    results = []
    for ctx in (32, 512):
        ids = torch.randint(1000, 20000, (1, ctx), generator=torch.Generator().manual_seed(0)).to(device)
        with torch.no_grad():
            model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)   # 예열
            t1 = time.time()
            out = model.generate(ids, max_new_tokens=64, min_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.pad_token_id, return_dict_in_generate=True)
            dt = time.time() - t1
        kv = out.past_key_values
        kv_real = sum(t.numel() * t.element_size() for layer in kv.layers for t in (layer.keys, layer.values))
        results.append({"ctx": ctx, "tok_per_s": 64 / dt, "kv_mib": kv_real / MiB,
                        "kv_formula_mib": kv_mib(cfg, ctx + 63, 4)})
        print(f"  문맥 {ctx:>3} + 생성 64: {64 / dt:5.1f}토큰/초, KV Cache 실제 {kv_real / MiB:5.2f}MiB"
              f" / 계산 {kv_mib(cfg, ctx + 63, 4):5.2f}MiB")
    print(f"  로딩 {load_s:.1f}초, 최대 메모리(프로세스) 로딩 후 {rss_loaded:.0f}MiB → 생성 후 {rss_mib():.0f}MiB")
    print(f"  측정 조건: {platform.machine()}, 스레드 {torch.get_num_threads()}개, PyTorch {torch.__version__}")
    OUT.write_text(json.dumps({"fits": table, "cpu": results, "load_s": load_s, "rss_mib": rss_mib(),
                               "threads": torch.get_num_threads()}, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")


if __name__ == "__main__":
    main()
