# 저자 GPU(예: RTX 3080 Ti)에서 교재의 미검증 항목을 한 번에 확인하고 보고서를 남긴다
#
# 실행(저장소 맨 위 폴더에서): python tools/gpu_verify.py
# 결과: reports/gpu_report.json, reports/gpu_report.md  ← 이 두 파일을 보내 주면 교재에 반영한다
#
# 단계마다 실패해도 멈추지 않고 다음 단계로 넘어가며, 실패 내용을 보고서에 적는다.
#   1. 환경 기록(GPU, 드라이버, 라이브러리 판)
#   2. 필요한 모델 다시 만들기(models/ 폴더에 없을 때만: 사전학습 Tiny → QLoRA → 최종 LoRA → 병합 → GGUF)
#   3. QLoRA 학습 메모리(제49장): GPU 메모리를 2GB로 제한해 GT 1030의 용량을 모사하고 최대 사용량 기록
#   4. Q8_0 GGUF 품질과 속도(제55장): GPU에서 PPL이 CPU 결과와 같은지, 초당 토큰
#   5. Ollama(제56장, 설치되어 있을 때): 등록 → validation 질문 6개 답을 제55장 Q8_0 답과 비교 → ollama ps
#   6. llama.cpp(설치되어 있을 때): llama-bench로 Q8_0·Q4_0 초당 토큰
# 2GB 제한은 "메모리 용량"만 모사한다. GT 1030의 속도와 Pascal 세대 특성(FP16 속도 등)은 모사하지 못한다.

import json
import math
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"
MODELS = ROOT / "models"
REPORT = ROOT / "reports"
GT1030_MIB = 2048
REBUILD = [                                       # (만들어지는 것, 실행할 실습)
    (MODELS / "qwen2-tiny-pretrained", "ch25/pretrain_tiny.py"),
    (MODELS / "qwen2-tiny-bike-lora", "ch30/peft_lora.py"),
    (MODELS / "ch49_qlora", "ch49/qlora_sft.py"),
    (MODELS / "ch50" / "원본" / "checkpoint-15", "ch50/improve.py"),
    (MODELS / "fpga_lora_v1", "ch51/save_load.py"),
    (MODELS / "fpga_merged_v1", "ch52/merge.py"),
    (MODELS / "gguf" / "fpga-q8_0.gguf", "ch55/quant_compare.py"),
    (MODELS / "gguf" / "Modelfile", "ch56/modelfile.py"),
]
result = {"steps": {}}


def step(name):
    """단계 하나를 실행하고 성공·실패·걸린 시간을 기록한다"""
    def wrap(fn):
        def run():
            t0 = time.time()
            print(f"\n=== {name} ===", flush=True)
            try:
                out = fn() or {}
                result["steps"][name] = {"상태": "성공", **out}
            except Exception as e:                                   # 실패해도 다음 단계로
                result["steps"][name] = {"상태": "실패", "오류": f"{type(e).__name__}: {e}",
                                         "자세히": traceback.format_exc()[-1500:]}
                print(f"  실패: {type(e).__name__}: {e}", flush=True)
            result["steps"][name]["초"] = round(time.time() - t0, 1)
            save()
        return run
    return wrap


def save():
    REPORT.mkdir(exist_ok=True)
    (REPORT / "gpu_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# GPU 확인 보고서", ""]
    for k, v in result.get("env", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    for name, v in result["steps"].items():
        lines.append(f"## {name}: {v['상태']} ({v.get('초', '')}초)")
        lines += [f"- {k}: {val}" for k, val in v.items() if k not in ("상태", "초", "자세히")]
        lines.append("")
    (REPORT / "gpu_report.md").write_text("\n".join(lines), encoding="utf-8")


def gpu_cap():
    import torch
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    torch.cuda.set_per_process_memory_fraction(min(1.0, GT1030_MIB / total), 0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return total


def environment():
    import torch
    env = {"운영체제": platform.platform(), "Python": platform.python_version(), "PyTorch": torch.__version__,
           "CUDA 사용 가능": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update({"GPU": p.name, "Compute Capability": f"{p.major}.{p.minor}",
                    "VRAM MiB": round(p.total_memory / 2 ** 20), "CUDA(PyTorch 빌드)": torch.version.cuda,
                    "지원 세대": " ".join(torch.cuda.get_arch_list())})
    smi = shutil.which("nvidia-smi")
    if smi:
        env["드라이버"] = subprocess.run([smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                                       capture_output=True, text=True).stdout.strip()
    from importlib.metadata import PackageNotFoundError, version
    for lib in ("transformers", "peft", "trl", "bitsandbytes", "gguf"):
        try:
            env[lib] = version(lib)
        except PackageNotFoundError:
            env[lib] = "설치 안 됨"
    env["Ollama"] = shutil.which("ollama") or "없음"
    env["llama-bench"] = shutil.which("llama-bench") or "없음"
    result["env"] = env
    for k, v in env.items():
        print(f"  {k}: {v}")


def snapshot() -> dict:
    return {p: p.read_bytes() for p in EX.rglob("*.json")}


@step("2. 모델 다시 만들기")
def rebuild():
    """없는 모델만 다시 만든다. 실습이 교재 기록(examples/의 .json)을 GPU 결과로 바꾸면
    그 파일은 reports/gpu_records/로 옮기고 교재 기록은 원래대로 되돌린다."""
    done, moved = [], []
    for target, script in REBUILD:
        if target.exists():
            continue
        print(f"  실행: examples/{script}", flush=True)
        before = snapshot()
        r = subprocess.run([sys.executable, str(EX / script)], cwd=ROOT, capture_output=True, text=True)
        for p, old in before.items():
            if p.exists() and p.read_bytes() != old:
                dest = REPORT / "gpu_records" / f"{p.parent.name}_{p.name}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(p.read_bytes())
                p.write_bytes(old)
                moved.append(str(dest.relative_to(ROOT)))
        if r.returncode != 0 or not target.exists():
            listing = sorted(str(x.relative_to(MODELS)) for x in target.parent.rglob("*"))[:40] \
                if target.parent.exists() else "폴더 없음"
            raise RuntimeError(f"{script} 뒤에 {target.relative_to(ROOT)} 없음(종료 코드 {r.returncode}). "
                               f"폴더 내용: {listing}. 오류: {r.stderr[-800:]}")
        done.append(script)
    return {"실행한 실습": done or "없음(모두 있음)", "GPU 결과로 따로 둔 기록": moved or "없음"}


@step("3. QLoRA 학습 메모리(2GB 제한 모사)")
def qlora_memory():
    import torch
    sys.path[:0] = [str(EX), str(EX / "ch49"), str(EX / "ch46"), str(EX / "ch47"), str(EX / "ch48")]
    from peft import get_peft_model
    from common.qwen import load_tokenizer
    from lora_sft import SCOPES, lora_config
    from qlora_sft import build
    from sft_loop import SFTDataset, split_rows
    total = gpu_cap()
    tok = load_tokenizer()
    tr = SFTDataset(tok, split_rows("train"))
    model, parts, _ = build("cuda")
    model = get_peft_model(model, lora_config(SCOPES["모든 Linear"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    longest = sorted(tr.items, key=lambda e: -len(e["input_ids"]))[:5]
    losses = []
    for e in longest:                                                 # 가장 긴 샘플 5개로 5단계
        ids = torch.tensor([e["input_ids"]], device="cuda")
        lab = torch.tensor([e["labels"]], device="cuda")
        loss = model(input_ids=ids, labels=lab).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(round(loss.item(), 4))
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    return {"GPU 전체 MiB": round(total), "제한 MiB": GT1030_MIB, "최대 할당 MiB": round(peak),
            "교재 계산값 MiB": 895.1, "손실": losses, "NaN·Inf 없음": all(math.isfinite(x) for x in losses)}


@step("4. Q8_0 GGUF 품질과 속도(GPU)")
def gguf_quality():
    import torch
    sys.path[:0] = [str(EX), str(EX / "ch46"), str(EX / "ch47")]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sft_loop import SFTDataset, split_rows, validate
    torch.cuda.set_per_process_memory_fraction(1.0, 0)            # Transformers는 GGUF를 FP32로 풀어 1.9GB를 쓰므로 제한을 푼다
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(MODELS / "fpga_merged_v1")
    model = AutoModelForCausalLM.from_pretrained(MODELS / "gguf", gguf_file="fpga-q8_0.gguf",
                                                 dtype=torch.float32).to("cuda").eval()
    ppl = validate(model, SFTDataset(tok, split_rows("validation")))["val_ppl"]
    book = json.loads((EX / "ch55" / "quant.json").read_text(encoding="utf-8"))["Q8_0"]["domain_ppl"]
    ids = tok("FPGA의 LUT는 어떤 역할을 하나요?", return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        model.generate(ids, max_new_tokens=4, do_sample=False)
        torch.cuda.synchronize()
        t0 = time.time()
        model.generate(ids, max_new_tokens=64, min_new_tokens=64, do_sample=False)
        torch.cuda.synchronize()
        dt = time.time() - t0
    return {"도메인 PPL(GPU)": round(ppl, 4), "교재(CPU) 값": round(book, 4),
            "차이 0.05 이내": abs(ppl - book) < 0.05, "초당 토큰(Transformers, FP32)": round(64 / dt, 1),
            "최대 할당 MiB(FP32로 푼 상태, 제한 없음)": round(torch.cuda.max_memory_allocated() / 2 ** 20)}


@step("5. Ollama 등록과 응답 비교")
def ollama():
    exe = shutil.which("ollama")
    if not exe:
        return {"결과": "Ollama 없음(건너뜀)"}
    sys.path[:0] = [str(EX / "ch57"), str(EX / "ch44"), str(EX / "ch46"), str(EX / "ch41"), str(EX)]
    from chat_template import SYSTEM
    from local_api import Client
    from sft_loop import split_rows
    r = subprocess.run([exe, "create", "fpga-qwen", "-f", "Modelfile"], cwd=MODELS / "gguf",
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ollama create 실패: {r.stderr[-500:]}")
    book = json.loads((EX / "ch55" / "quant.json").read_text(encoding="utf-8"))["Q8_0"]["answers"]
    client = Client(model="fpga-qwen", timeout=300)
    same, rows = 0, {}
    for row in split_rows("validation"):
        q = row["instruction"] + (f"\n\n{row['input']}" if row["input"] else "")
        ans = client.generate(q, system=SYSTEM).strip()
        rows[row["id"]] = ans[:80]
        same += ans == book[row["id"]].strip()
    ps = subprocess.run([exe, "ps"], capture_output=True, text=True).stdout.strip()
    return {"교재 Q8_0 답과 완전히 같은 수": f"{same}/6", "답 앞부분": rows, "ollama ps": ps}


@step("6. llama.cpp 속도")
def llama_bench():
    exe = shutil.which("llama-bench")
    if not exe:
        return {"결과": "llama-bench 없음(건너뜀)"}
    out = {}
    for f in ("fpga-q8_0.gguf", "fpga-q4_0.gguf"):
        r = subprocess.run([exe, "-m", str(MODELS / "gguf" / f), "-ngl", "99"], capture_output=True, text=True)
        out[f] = r.stdout[-1200:] if r.returncode == 0 else f"실패: {r.stderr[-400:]}"
    return out


if __name__ == "__main__":
    print("=== 1. 환경 ===")
    environment()
    save()
    rebuild()
    import torch
    if torch.cuda.is_available():
        qlora_memory()
        torch.cuda.empty_cache()
        gguf_quality()
        torch.cuda.empty_cache()
    else:
        result["steps"]["GPU 단계"] = {"상태": "건너뜀", "이유": "CUDA를 쓸 수 없음(드라이버·PyTorch 빌드 확인)"}
    ollama()
    llama_bench()
    save()
    print(f"\n보고서: {REPORT / 'gpu_report.md'}")
