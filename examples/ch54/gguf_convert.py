# 제54장 핵심 실습: Qwen2-Tiny와 0.5B 병합 모델을 GGUF로 바꾸고, 다시 읽어 Logits·Greedy 출력을 비교한다
#
# 실행: python examples/ch54/gguf_convert.py   (CPU 약 3분, 메모리 8GB에서 실행)
#
# 만드는 파일: models/gguf/(tiny-f32, fpga-f32, fpga-f16 .gguf), examples/ch54/compare.json
# 변환한 GGUF를 llama.cpp로 실행해 보는 일은 교재 검증 환경에서 하지 않았다(미검증).
# 여기서는 Transformers의 GGUF 읽기 기능으로 파일을 다시 불러와 원본과 비교한다.

import gc
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, HERE.parent, HERE.parent / "ch41", HERE.parent / "ch44"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.utils import logging  # noqa: E402

from chat_template import SYSTEM, render  # noqa: E402            (제44장)
from check_dataset import load  # noqa: E402                     (제41장)
from common.device import select_device  # noqa: E402
from to_gguf import convert, read_summary  # noqa: E402

logging.disable_progress_bar()
logging.set_verbosity_error()
OUT_DIR = ROOT / "models" / "gguf"
JOBS = [("tiny-f32", ROOT / "models" / "qwen2-tiny-pretrained", "F32"),
        ("fpga-f32", ROOT / "models" / "fpga_merged_v1", "F32"),
        ("fpga-f16", ROOT / "models" / "fpga_merged_v1", "F16")]
MiB = 2 ** 20


def gen_config(src: Path):
    """원본 폴더의 생성 설정(반복 억제 1.1 등)을 그대로 쓰되 Greedy로. GGUF에는 이 파일이 들어가지 않는다."""
    from transformers import GenerationConfig
    try:
        g = GenerationConfig.from_pretrained(src)
    except OSError:
        g = GenerationConfig()
    g.do_sample, g.max_new_tokens, g.pad_token_id = False, 24, 151643
    g.temperature = g.top_p = g.top_k = None
    return g


def greedy(model, ids, gen):
    with torch.no_grad():
        return model.generate(ids, generation_config=gen)[0, ids.shape[1]:]


def main() -> None:
    device, _ = select_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("── 1. 변환 ──")
    for name, src, ftype in JOBS:
        info = convert(src, OUT_DIR / f"{name}.gguf", ftype)
        src_mib = sum(f.stat().st_size for f in src.glob("*.safetensors")) / MiB
        print(f"  {name}: 텐서 {info['tensors']}개(LM Head 묶임 {info['tied']}), 자료형 {info['types']}, "
              f"Safetensors {src_mib:.0f}MiB → GGUF {(OUT_DIR / f'{name}.gguf').stat().st_size / MiB:.0f}MiB")

    print("\n── 2. GGUF 구조(fpga-f16) ──")
    s = read_summary(OUT_DIR / "fpga-f16.gguf")
    print(f"  메타데이터 {s['n_fields']}개, 텐서 {s['n_tensors']}개, Chat Template 포함 {s['has_template']}")
    for k, v in s["kv"].items():
        print(f"    {k} = {v}")
    print(f"  첫 텐서 {s['first'][0]}: 모양 {s['first'][1]}(GGUF는 차원 순서를 거꾸로 적는다), {s['first'][2]}")

    print("\n── 3. 같은 Token ID로 Logits 비교(원본 Safetensors ↔ GGUF를 다시 읽은 모델) ──")
    rows = load()
    results = {}
    for name, src, _ in JOBS:
        tok = AutoTokenizer.from_pretrained(src)
        tok_g = AutoTokenizer.from_pretrained(OUT_DIR, gguf_file=f"{name}.gguf")
        texts = [render(tok, [{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": r["instruction"]}], gen=True) for r in rows[:10]]
        same_ids = all(tok(t).input_ids == tok_g(t).input_ids for t in texts)
        ids = torch.tensor([tok(texts[2]).input_ids], device=device)
        a = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).to(device).eval()
        with torch.no_grad():
            la = a(ids).logits.float().cpu()
        gen = gen_config(src)
        ga = greedy(a, ids, gen)
        del a
        gc.collect()
        b = AutoModelForCausalLM.from_pretrained(OUT_DIR, gguf_file=f"{name}.gguf", dtype=torch.float32).to(device).eval()
        with torch.no_grad():
            lb = b(ids).logits.float().cpu()
        gb = greedy(b, ids, gen)
        plain = gen_config(ROOT / "없는 폴더")                               # 생성 설정 파일이 없을 때(반복 억제 없음)
        gb_default = greedy(b, ids, plain) if name == "fpga-f32" else gb
        del b
        gc.collect()
        d = (la - lb).abs().max().item()
        same_greedy = torch.equal(ga, gb)
        results[name] = {"token_ids_same": same_ids, "logits_max_diff": d, "greedy_same": same_greedy,
                         "repetition_penalty": gen.repetition_penalty}
        print(f"  {name}: 토큰화 10문장 같음 {same_ids}, Logits 최대 차이 {d:.2e}, Greedy 24토큰 같음 {same_greedy}")
        if name == "fpga-f32":
            results[name]["greedy_same_without_config"] = torch.equal(ga, gb_default)
            print(f"    생성 설정(반복 억제 {gen.repetition_penalty})을 옮기지 않으면 Greedy 같음 "
                  f"{torch.equal(ga, gb_default)}")
    (HERE / "compare.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
